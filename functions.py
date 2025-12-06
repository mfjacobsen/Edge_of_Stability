import seed

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

device = seed.device
generator = seed.generator

def sample_data(X, y, num_per_class):
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    indices = []

    for c in classes:
        cls_idx = np.where(y == c)[0]
        chosen = np.random.choice(cls_idx, num_per_class, replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    return X[indices], y[indices]

def load_cifar_10(num_per_class=500, test_num_per_class=100):
    # Load raw CIFAR-10 
    train = datasets.CIFAR10(root="./data", train=True,  download=True)
    test  = datasets.CIFAR10(root="./data", train=False, download=True)

    # # Subsample
    X, y  = sample_data(train.data, train.targets, num_per_class)
    X_test, y_test = sample_data(test.data, test.targets, test_num_per_class)

    # Convert to float and scale
    X  = X.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Normalize
    mean = X.mean(axis=(0,1,2),keepdims=True)
    std = X.std(axis=(0,1,2),keepdims=True)
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    # Reshape to NCHW
    X = np.transpose(X, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # Convert to torch
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    # One hot encode labels for RMSE criterion
    y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float().to(device)
    y_test_onehot = torch.nn.functional.one_hot(y_test, num_classes=10).float().to(device)

    return X, y, X_test, y_test, y_onehot, y_test_onehot

def setup_output_files(output_dir="output"): 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata_path = os.path.join(output_dir, "metadata_replication.csv")
    output_data_path = os.path.join(output_dir, "output_replication.csv")

    if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
    else:
        metadata = pd.DataFrame({
            "model_id": pd.Series(dtype="int"),
            "model_type": pd.Series(dtype="str"),
            "activation_function": pd.Series(dtype="str"),
            "optimizer": pd.Series(dtype="str"),
            "criterion": pd.Series(dtype="str"),
            "learning_rate": pd.Series(dtype="float"),
            "momentum": pd.Series(dtype="float"),
            "num_epochs": pd.Series(dtype="int"),
            "time_minutes": pd.Series(dtype="float"),
        })

    if os.path.exists(output_data_path):
        output_data = pd.read_csv(output_data_path)
    else:
        output_data = pd.DataFrame({
            "model_id": pd.Series(dtype="int"),
            "epoch": pd.Series(dtype="int"),
            "train_loss": pd.Series(dtype="float"),
            "train_accuracy": pd.Series(dtype="float"),
            "test_accuracy": pd.Series(dtype="float"),
            "sharpness_H": pd.Series(dtype="float"),
            "sharpness_A": pd.Series(dtype="float"),
        })

    return metadata, output_data

def load_output_files(output_dir="output"):
    metadata_path = os.path.join(output_dir, "metadata_replication.csv")
    output_data_path = os.path.join(output_dir, "output_replication.csv")

    metadata = pd.read_csv(metadata_path)
    output_data = pd.read_csv(output_data_path)

    return metadata, output_data

def save_output_files(metadata, output_data, output_dir="output"):

    metadata_path = os.path.join(output_dir, "metadata_replication.csv")
    output_data_path = os.path.join(output_dir, "output_replication.csv")

    metadata.to_csv(metadata_path, index=False)
    output_data.to_csv(output_data_path, index=False)

def delete_model_data(model_ids, output_dir="output"):
    metadata, output_data = load_output_files(output_dir)
    metadata = metadata[~metadata['model_id'].isin(model_ids)]
    output_data = output_data[~output_data['model_id'].isin(model_ids)]
    save_output_files(metadata, output_data, output_dir)

def get_hessian_metrics(model, optimizer, criterion, X, y, 
                        subsample_dim = 1024, iters=30, tol = 1e-4):
    
    # Subsample data for compute efficiency
    subsample_dim = min(subsample_dim, len(X))
    idx = torch.randperm(len(X), device=X.device, generator=generator)[:subsample_dim]
    X = X[idx]
    y = y[idx]
    
    # Build graph for gradient
    outputs = model(X)
    loss = criterion(outputs, y)

    grads = torch.autograd.grad(
        loss, model.param_list,
        create_graph=True
    )
    g_flat = torch.cat([g.reshape(-1) for g in grads])
    dim    = g_flat.numel()
    device = g_flat.device

    # Computes Hessian-vector product with Pearlmutter trick
    def Hv(v):
        Hv_list = torch.autograd.grad(
            g_flat @ v,
            model.param_list,
            retain_graph=True
        )
        return torch.cat([h.reshape(-1) for h in Hv_list])
    
    # Performs power iteration to estimate largest eigenvalue
    def power_iteration(matvec):
        v = torch.randn(dim, device=device, generator=generator)
        v /= v.norm()

        eig_old = 0.0
        for _ in range(iters):
            Hv_v = matvec(v)
            eig = (v @ Hv_v).item()   
            v = Hv_v / Hv_v.norm()

            if abs(eig - eig_old) / (abs(eig_old) + 1e-12) < tol:
                break
            eig_old = eig

        Hv_v = matvec(v)
        eig = (v @ Hv_v).item()
        return eig

    lambda_H = power_iteration(Hv)
    
    if isinstance(optimizer, torch.optim.RMSprop):
        
        # Compute adaptive scaling matrix D (sqrt) for effective Hessian
        v_t = torch.cat([state['square_avg'].reshape(-1)
                        for state in optimizer.state.values()]
                        ).detach()

        eps = optimizer.param_groups[0]['eps']
        D_sqrt = torch.sqrt(1 / torch.sqrt(v_t + eps))

        # Compute effective Hessian-vector product
        def Av(v):
            return D_sqrt * Hv(D_sqrt * v)
        
        lambda_A = power_iteration(Av)
    else:
        lambda_A = None

    return lambda_H, lambda_A

def train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test):
    print(f"Training {model.__class__.__name__} with " +
          f"{optimizer.__class__.__name__} and learning rate " +
          f"{optimizer.param_groups[0]['lr']} for {epochs} epochs.")

    learning_rate = optimizer.param_groups[0]['lr']
    momentum = optimizer.param_groups[0].get('momentum', 0.0)

    model.to(device)
    model.train()

    train_losses = np.full(epochs, np.nan)
    train_accuracies = np.full(epochs, np.nan)
    test_accuracies = np.full(epochs, np.nan)
    H_sharps = np.full(epochs, np.nan)
    A_sharps = np.full(epochs, np.nan)

    if isinstance(criterion, nn.MSELoss):
        y_loss = torch.nn.functional.one_hot(
            y, num_classes=model.num_labels).float().to(device)
       
    else:
        y_loss = y.to(device)

    start = time.time()
    
    train_acc = 0.0
    epoch = 0

    while train_acc < accuracy and epoch < epochs :

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y_loss)
        loss.backward()
        optimizer.step()

        train_losses[epoch] = loss.item()

        if epoch % (epochs // 100) == 0:
            H_sharps[epoch], A_sharps[epoch] = get_hessian_metrics(
                model, optimizer, criterion, X, y_loss
            )

        with torch.no_grad():
            model.eval()
            train_preds = outputs.argmax(dim=1)
            test_preds = model(X_test).argmax(dim=1)
            train_acc = (train_preds == y).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()
            train_accuracies[epoch] = train_acc
            test_accuracies[epoch] = test_acc
        model.train()

        if (epoch+1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, " +
                  f"Time: {round(((time.time() - start) / 60), 2)}, " +
                  f"Train Acc: {train_accuracies[epoch]:.4f}, " +
                  f"Test Acc: {test_accuracies[epoch]:.4f}, ")
        epoch += 1

    metadata, output_data = setup_output_files("output")
    model_id = metadata.shape[0] + 1

    metadata.loc[metadata.shape[0]] ={
        "model_id": model_id,
        "model_type": model.__class__.__name__,
        "activation_function": model.activation.__name__,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "num_epochs": epochs,
        "time_minutes": round((time.time() - start) / 60, 2),
    }

    output_data = pd.concat([output_data, pd.DataFrame({
        "model_id": np.ones_like(train_losses) * model_id,
        "epoch": np.arange(1, epochs + 1),
        "train_loss": train_losses,
        "sharpness_H": H_sharps.round(4),
        "sharpness_A": A_sharps.round(4),
        "test_accuracy": test_accuracies,
        "train_accuracy": train_accuracies,
    })], ignore_index=True)

    save_output_files(metadata, output_data)

def plot_output_data(metadata, output, model_id):
    metadata = metadata[metadata['model_id']==model_id]
    output = output[output['model_id']==model_id]
    
    xs = np.arange(metadata['num_epochs'].iloc[0])
    losses = output['train_loss']
    sharpness_H = output['sharpness_H']
    sharpness_A = output['sharpness_A']
    train_accuracy = output['train_accuracy']
    test_accuracy = output['test_accuracy']
    momentum = metadata['momentum'].iloc[0]
    learning_rate = metadata['learning_rate'].iloc[0]
    sharpness_H_lim = 2 * (1 + momentum) / learning_rate

    fig = make_subplots(rows = 2, cols = 1, 
                        specs=[[{"secondary_y": True}],
                               [{"secondary_y": True}]],
                        shared_xaxes=True,
                        vertical_spacing=0.1)
    
    fig.add_trace(
        go.Scatter(x=xs, y=losses, name="Training Loss",line=dict(width=2)),
        secondary_y=False, row=1, col=1
    )

    # fig.add_trace(
    #     go.Scatter(x=xs, y=sharpness_H, name="Max Eigenvalue of H", mode='markers', line=dict(width=2)),
    #     secondary_y=True, row=1, col=1
    # )

    fig.add_trace(
        go.Scatter(x=xs, y=sharpness_A, name="Max Eigenvalue of A", mode='markers', line=dict(width=2)),
        secondary_y=True, row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=xs, y=test_accuracy, name="Test Accuracy", line=dict(width=2)),
        secondary_y=False, row=2, col=1
    )

    fig.add_hline(y=2, line_dash="dash", line_color="black", 
                  row=1, col=1, secondary_y=True)

    

    fig.update_yaxes(title_text="Training Loss", secondary_y=False, 
                     range = [0,0.5], showgrid=False,
                     row=1, col=1)
    fig.update_yaxes(title_text="Max Eigenvalue of A", secondary_y=True, 
                     range = [0, 5],
                     row=1, col=1)
    
    fig.update_xaxes(title_text="epoch",
                     range = [0,5000])
    fig.update_layout(height = 1000, width = 1000)
    
    fig.show()

def plot_sgd_fcnn_data(metadata, output, model_ids_mse, model_ids_ce, save=True):

    max_epoch_mse = (
        output
        [(output["train_loss"].notna()) & (output["model_id"].isin(model_ids_mse))]
        ["epoch"]
        .max()
    )
    xs_mse = np.arange(max_epoch_mse)

    max_epoch_ce = (
        output
        [(output["train_loss"].notna()) & (output["model_id"].isin(model_ids_ce))]
        ["epoch"]
        .max()
    )
    xs_ce = np.arange(max_epoch_ce)

    fig = make_subplots(rows = 2, cols = 2, 
                        vertical_spacing=0.1, shared_xaxes=True,
                        subplot_titles=["MSE Loss", "Cross-Entropy Loss"] )
    colors = px.colors.qualitative.D3[:3]

    for i, model_id in enumerate(model_ids_mse):
        md = metadata[metadata['model_id']==model_id]
        out = output[output['model_id']==model_id]
        lr = md['learning_rate'].iloc[0]
        
        losses = out['train_loss']
        sharpness_H = out['sharpness_H']
    
        sharpness_H_lim = 2 / lr
        
        fig.add_trace(
            go.Scatter(x=xs_mse, y=losses, name= f"η = {lr}",
                       line=dict(width=2.5), marker_color=colors[i],
                       legend="legend",
                       showlegend=True), 
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=xs_mse, y=sharpness_H, name= "Sharpness of H", 
                       mode='markers', showlegend=False,
                       marker=dict(size=5), marker_color=colors[i]),
            row=2, col=1
        )

        fig.add_hline(y=sharpness_H_lim, line_dash="dash", line_color=colors[i], 
                        row=2, col=1)
        
    for i, model_id in enumerate(model_ids_ce):
        md = metadata[metadata['model_id']==model_id]
        out = output[output['model_id']==model_id]
        lr = md['learning_rate'].iloc[0]
        
        losses = out['train_loss']
        sharpness_H = out['sharpness_H']
    
        sharpness_H_lim = 2 / lr
        
        fig.add_trace(
            go.Scatter(x=xs_ce, y=losses, name= f"η = {lr}",
                       line=dict(width=2.5), marker_color=colors[i],
                       legend="legend2",
                       showlegend=True), 
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=xs_ce, y=sharpness_H, name= "Sharpness of H", 
                       mode='markers', showlegend=False,
                       marker=dict(size=5), marker_color=colors[i]),
            row=2, col=2
        )

        fig.add_hline(y=sharpness_H_lim, line_dash="dash", line_color=colors[i], 
                        row=2, col=2)
        
    mse_y_sharp_max = 2 / metadata[metadata["model_id"]==model_ids_mse[-1]]["learning_rate"].iloc[0]*1.1
    ce_y_sharp_max = 2 / metadata[metadata["model_id"]==model_ids_ce[-1]]["learning_rate"].iloc[0]*1.2


    fig.update_yaxes(title_text="Training Loss",
                    range = [0,0.08],
                    row=1, col=1)
    fig.update_yaxes(title_text="Sharpness",
                    range = [10, mse_y_sharp_max],
                    row=2, col=1)
    fig.update_yaxes(title_text="",
                    range = [0,1.5],
                    row=1, col=2)
    fig.update_yaxes(title_text="",
                    range = [20, ce_y_sharp_max],
                    row=2, col=2)
    
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)

    fig.update_layout(height = 400, width = 800, 
                      title = dict(text=f"FCNN with GD on CIFAR-10", x = 0.5),
                      legend=dict(x=0.29, y=0.99,
                                  bgcolor='rgba(255, 255, 255, 0.3)'),
                      legend2=dict(x=0.83, y=0.99,
                                   bgcolor='rgba(255, 255, 255, 0.3)')
                    )
    if save:
        fig.write_html("output/images/gd_fcnn_cifar10.html",
                    width = 800, height = 400, scale = 4)
    fig.show()

def plot_sgdm_fcnn_data(metadata, output, model_ids_mse, model_ids_ce, save=True):

    max_epoch_mse = (
        output
        [(output["train_loss"].notna()) & (output["model_id"].isin(model_ids_mse))]
        ["epoch"]
        .max()
    )
    xs_mse = np.arange(max_epoch_mse)

    max_epoch_ce = (
        output
        [(output["train_loss"].notna()) & (output["model_id"].isin(model_ids_ce))]
        ["epoch"]
        .max()
    )
    xs_ce = np.arange(max_epoch_ce)

    fig = make_subplots(rows = 2, cols = 2, 
                        vertical_spacing=0.1, shared_xaxes=True,
                        subplot_titles=["MSE Loss", "Cross-Entropy Loss"] )
    colors = px.colors.qualitative.D3[:3]

    for i, model_id in enumerate(model_ids_mse):
        md = metadata[metadata['model_id']==model_id]
        out = output[output['model_id']==model_id]
        lr = md['learning_rate'].iloc[0]
        momentum = md['momentum'].iloc[0]
        
        losses = out['train_loss']
        sharpness_H = out['sharpness_H']
        
        sharpness_H_lim = 2 * (1 + momentum) / lr
        
        fig.add_trace(
            go.Scatter(x=xs_mse, y=losses, name= f"η = {lr}",
                       line=dict(width=2.5), marker_color=colors[i],
                       legend="legend",
                       showlegend=True), 
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=xs_mse, y=sharpness_H, name= "Sharpness of H", 
                       mode='markers', showlegend=False,
                       marker=dict(size=5), marker_color=colors[i]),
            row=2, col=1
        )

        fig.add_hline(y=sharpness_H_lim, line_dash="dash", line_color=colors[i], 
                        row=2, col=1)
        
    for i, model_id in enumerate(model_ids_ce):
        md = metadata[metadata['model_id']==model_id]
        out = output[output['model_id']==model_id]
        lr = md['learning_rate'].iloc[0]
        momentum = md['momentum'].iloc[0]
        
        losses = out['train_loss']
        sharpness_H = out['sharpness_H']
    
        sharpness_H_lim = 2 * (1 + momentum) / lr
        
        fig.add_trace(
            go.Scatter(x=xs_ce, y=losses, name= f"η = {lr}",
                       line=dict(width=2.5), marker_color=colors[i],
                       legend="legend2",
                       showlegend=True), 
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=xs_ce, y=sharpness_H, name= "Sharpness of H", 
                       mode='markers', showlegend=False,
                       marker=dict(size=5), marker_color=colors[i]),
            row=2, col=2
        )

        fig.add_hline(y=sharpness_H_lim, line_dash="dash", line_color=colors[i], 
                        row=2, col=2)
        
    mse_y_sharp_max = (
        2 * (1 + metadata[metadata["model_id"]==model_ids_mse[-1]]
                         ["momentum"].iloc[0]) 
          / metadata[metadata["model_id"]==model_ids_mse[-1]]
                    ["learning_rate"].iloc[0] 
          * 1.1
    )

    ce_y_sharp_max = (
        2 * (1 + metadata[metadata["model_id"]==model_ids_ce[-1]]
                         ["momentum"].iloc[0]) 
          / metadata[metadata["model_id"]==model_ids_ce[-1]]
                    ["learning_rate"].iloc[0]
          *1.2
    )


    fig.update_yaxes(title_text="Training Loss",
                    range = [0,0.08],
                    row=1, col=1)
    fig.update_yaxes(title_text="Sharpness",
                    range = [10, mse_y_sharp_max],
                    row=2, col=1)
    fig.update_yaxes(title_text="",
                    range = [0,2],
                    row=1, col=2)
    fig.update_yaxes(title_text="",
                    range = [0, ce_y_sharp_max],
                    row=2, col=2)
    
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)

    fig.update_layout(height = 400, width = 800, 
                      title = dict(text=f"FCNN with GD and Momentum on CIFAR-10", x = 0.5),
                      legend=dict(x=0.29, y=0.99,
                                  bgcolor='rgba(255, 255, 255, 0.3)'),
                      legend2=dict(x=0.83, y=0.99,
                                   bgcolor='rgba(255, 255, 255, 0.3)')
                    )
    if save:
        fig.write_html("output/images/gd_mom_fcnn_cifar10.html",
                    width = 800, height = 400, scale = 4)
    fig.show()

def plot_rmsprop_fcnn_data(metadata, output, model_ids_mse, model_ids_ce, save=True):

    max_epoch_mse = (
        output
        [(output["train_loss"].notna()) & (output["model_id"].isin(model_ids_mse))]
        ["epoch"]
        .max()
    )
    xs_mse = np.arange(max_epoch_mse)

    max_epoch_ce = (
        output
        [(output["train_loss"].notna()) & (output["model_id"].isin(model_ids_ce))]
        ["epoch"]
        .max()
    )
    xs_ce = np.arange(max_epoch_ce)

    fig = make_subplots(rows = 2, cols = 2, 
                        vertical_spacing=0.1, shared_xaxes=True,
                        subplot_titles=["MSE Loss", "Cross-Entropy Loss"] )
    colors = px.colors.qualitative.D3[:3]

    for i, model_id in enumerate(model_ids_mse):
        md = metadata[metadata['model_id']==model_id]
        out = output[output['model_id']==model_id]
        lr = md['learning_rate'].iloc[0]
        
        losses = out['train_loss']
        sharpness_H = out['sharpness_A']
        
        sharpness_H_lim = 2 / lr
        
        fig.add_trace(
            go.Scatter(x=xs_mse, y=losses, name= f"η = {lr}",
                       line=dict(width=2.5), marker_color=colors[i],
                       legend="legend",
                       showlegend=True), 
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=xs_mse, y=sharpness_H, name= "Sharpness of Effective Hessian", 
                       mode='markers', showlegend=False,
                       marker=dict(size=5), marker_color=colors[i]),
            row=2, col=1
        )

        fig.add_hline(y=sharpness_H_lim, line_dash="dash", line_color=colors[i], 
                        row=2, col=1)
        
    for i, model_id in enumerate(model_ids_ce):
        md = metadata[metadata['model_id']==model_id]
        out = output[output['model_id']==model_id]
        lr = md['learning_rate'].iloc[0]
        
        losses = out['train_loss']
        sharpness_H = out['sharpness_A']
    
        sharpness_H_lim = 2 / lr
        
        fig.add_trace(
            go.Scatter(x=xs_ce, y=losses, name= f"η = {lr}",
                       line=dict(width=2.5), marker_color=colors[i],
                       legend="legend2",
                       showlegend=True), 
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=xs_ce, y=sharpness_H, name= "Sharpness of Effective Hessian", 
                       mode='markers', showlegend=False,
                       marker=dict(size=5), marker_color=colors[i]),
            row=2, col=2
        )

        fig.add_hline(y=sharpness_H_lim, line_dash="dash", line_color=colors[i], 
                        row=2, col=2)

    mse_y_sharp_max = (
        2 * (1 + metadata[metadata["model_id"]==model_ids_mse[-1]]
                         ["momentum"].iloc[0]) 
          / metadata[metadata["model_id"]==model_ids_mse[-1]]
                    ["learning_rate"].iloc[0] 
          * 1.1
    )

    ce_y_sharp_max = (
        2 * (1 + metadata[metadata["model_id"]==model_ids_ce[-1]]
                         ["momentum"].iloc[0]) 
          / metadata[metadata["model_id"]==model_ids_ce[-1]]
                    ["learning_rate"].iloc[0]
          *1.2
    )

    fig.update_yaxes(title_text="Training Loss",
                    range = [0.01, 0.11],
                    row=1, col=1)
    fig.update_yaxes(title_text="Sharpness",
                    range = [0, mse_y_sharp_max],
                    row=2, col=1)
    fig.update_yaxes(title_text="",
                    range = [0,1.75],
                    row=1, col=2)
    fig.update_yaxes(title_text="",
                    range = [0, ce_y_sharp_max],
                    row=2, col=2)
    
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)

    fig.update_layout(height = 400, width = 800, 
                      title = dict(text=f"FCNN with RMSProp on CIFAR-10", x = 0.5),
                      legend=dict(x=0.29, y=0.99,
                                  bgcolor='rgba(255, 255, 255, 0.3)'),
                      legend2=dict(x=0.83, y=0.99,
                                   bgcolor='rgba(255, 255, 255, 0.3)')
                    )
    if save:
        fig.write_html("output/images/rmsprop_fcnn_cifar10.html",
                    width = 800, height = 400, scale = 4)
    fig.show()

def generate_gd_quadratic_plot():
    A = np.array([[1, 1],
                [1, 8]])

    def f(x, y):
        X = np.array([x, y])
        return 0.5 * X.T @ A @ X

    def grad(x):
        return A @ x

    lambda_max = np.linalg.eigvalsh(A).max()

    eta_conv = 1.8 / lambda_max
    eta_div = 2.05 / lambda_max
    steps = 20

    xs_conv = []
    xs_div = []
    x_conv = np.array([-2.5, 1.5])
    x_div = np.array([-2.5, 1.5])

    for _ in range(steps):
        xs_conv.append(x_conv.copy())
        xs_div.append(x_div.copy())
        x_conv = x_conv - eta_conv * grad(x_conv)
        x_div = x_div - eta_div * grad(x_div)

    xs_conv = np.array(xs_conv)
    xs_div = np.array(xs_div)

    gx = np.linspace(-3, 3, 200)
    gy = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(gx, gy)
    Z = 0.5*(A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2)

    fig = make_subplots(rows = 1, cols = 2, horizontal_spacing=0.05,
                        subplot_titles=("η < 2 / λ_max", "η > 2 / λ_max"))

    fig.add_trace(go.Contour(
        x=gx, y=gy, z=Z,
        contours=dict(
            coloring="lines",
            showlabels=False
        ),
        line_width=1,
        colorscale="Viridis",
        showscale=False
    ), row=1, col=1)

    fig.add_trace(go.Contour(
        x=gx, y=gy, z=Z,
        contours=dict(
            coloring="lines",
            showlabels=False
        ),
        line_width=1,
        colorscale="Viridis",
        showscale=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=xs_conv[:,0], y=xs_conv[:,1],
        mode="lines+markers",
        line=dict(width=2, color="red"),
        marker=dict(size=5, color="red"),
        name="GD Path"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=xs_div[:,0], y=xs_div[:,1],
        mode="lines+markers",
        line=dict(width=2, color="red"),
        marker=dict(size=5, color="red"),
        name="GD Path"
    ), row=1, col=2)

    fig.update_yaxes(showticklabels=True, ticks="", row=1, col=1)
    fig.update_yaxes(showticklabels=False, ticks="", row=1, col=2)

    fig.update_layout(
        title=dict(text="Gradient Descent on a Quadratic", x =0.5),
        xaxis1_title="x₁",
        yaxis1_title="x₂",
        xaxis2_title="x₁",
        width=600,
        height=300,
        showlegend=False,
        margin=dict(l=15, r=60, t=80, b=30)
    )

    fig.show()
    fig.write_html("output/images/gd_quadratic.html",
                    width = 600, height = 300, scale = 4)