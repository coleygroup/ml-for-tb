# tbprop
TB properties using deep learning

### Setup

1. [Install Miniconda.](https://docs.anaconda.com/free/miniconda/miniconda-install/)
2. Clone `tbprop`.
```bash
git clone git@github.com:freundjs/tbprop.git
```
3. Create conda environment. Install the version of PyTorch that best suits your system from [here](https://pytorch.org/get-started/locally/). Example installations for Mac (MPS) and Linux (GPU) are given below.
```bash
cd /path/to/tbprop
conda env create -f environment.yml
conda activate tbprop

# Install Pytorch. E.g.
conda install pytorch -c pytorch-nightly # for Mac with MPS OR
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia # for Linux with GPU (CUDA version 12.1)
```
4. Download the data and place the `data` directory in the project directory. *Link to data to be posted soon!*

### Running the code

1. Before running any code, you have to navigate to the project directory and set the `PYTHONPATH` as follows.
```bash
cd /path/to/tbprop
conda activate tbprop
export PYTHONPATH=. # Or add this path to PYTHONPATH in ~/.zshrc
```
2. Optimize the hyperparameters for a given deep learning algorithm:
    - Set the model name and other settings in `sh_scripts/optimize_model.sh`.
    - Execute the script.
```bash
bash sh_scripts/optimize_model.sh
```
3. Train a model.
    - Set the various parameters in `sh_scripts/train_model.sh`.
    - Execute the script.
```bash
bash sh_scripts/train_model.sh
```