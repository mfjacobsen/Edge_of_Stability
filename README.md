## Edge of Stability

This repository runs code that generates empirical evidence of the edge of stability in neural networks.

The repository has the following structure:

- **data** - Contains data (CIFAR-10) that will be downloaded when the project is first run.
- **notebooks** - contains exploratory analysis and replication notebooks.
- **src** - source code to run the project.
- **output** - output data and images.

- environment.yml - conda environment 
- README.md

### Steps for running:
```
git clone https://github.com/mfjacobsen/Edge_of_Stability
cd Edge_of_Stability
conda env create -f environment.yml
conda activate stability-env
python -m src.run
```
These steps will automatically download the data, set up the conda environment, and run the project code. 

Output images are automatically displayed in the web broweser and stored in output/images. 

Note, due to pytorch implementation, this code is not completely deterministic. Small variations in output images are normal, however, the experiments are designed to consistently show the edge of stability.
