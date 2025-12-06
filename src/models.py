from . import seed

import torch
import torch.nn as nn

SEED = seed.SEED

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, 
                 num_labels, activation):
        super(FullyConnectedNet, self).__init__()

        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers_size = hidden_layer_size
        self.num_labels = num_labels
        self.activation = activation

        layers = [nn.Flatten()]

        in_size = input_size
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(in_size, hidden_layer_size), activation()]
            in_size = hidden_layer_size

        layers.append(nn.Linear(in_size, num_labels))
        self.network = nn.Sequential(*layers)
        self.param_list = list(self.parameters())

        for m in self.network:
            if isinstance(m, nn.Linear):
                if activation == nn.ReLU:
                    nn.init.kaiming_normal_(
                        m.weight, 
                        generator=torch.Generator().manual_seed(SEED), 
                        nonlinearity='relu')
                elif activation == nn.Tanh:
                    nn.init.xavier_uniform_(
                        m.weight,
                        generator=torch.Generator().manual_seed(SEED)
                        )
                else:
                    nn.init.kaiming_normal_(
                        m.weight, 
                        generator=torch.Generator().manual_seed(SEED), 
                        nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)
    
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.activation = nn.ReLU

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        self.param_list = list(self.parameters())

    def forward(self, x):
        return self.classifier(self.features(x))