# Machine Learning Models for Molecular Property Predictions for TB Drug Discovery

## Environment Setup

1. [Install Miniconda.](https://docs.anaconda.com/free/miniconda/miniconda-install/)
2. Create conda environment. Install the version of PyTorch that best suits your system from [here](https://pytorch.org/get-started/locally/). Example installations for Mac (MPS) and Linux (GPU) are given below.
```bash
cd /path/to/tbprop
conda env create -f environment.yml
conda activate tbprop

# Install Pytorch. E.g.
conda install pytorch -c pytorch-nightly # for Mac with MPS OR
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia # for Linux with GPU (CUDA version 12.1)
```
4. Download the data and place the `data` directory in the project directory. *Link to data to be posted soon!*

## Running The Code

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

### MIC Models

Our efforts to build the MIC models are contained in the notebooks in `./mic_models`. These include notebooks which take the original datasets (TAACF-CB2, TAACF-SRIKinase, and MLSMR), curate the data, and build models. Note that these were then merged into two datasets - TAACF and MLSMR.

The train, val, test split for the merged MIC datasets is available in `ml-for-tb/data/0_raw.tar.xz` and you can also use the `tbprop` functionality to train them. You can follow `./notebooks/classification_tutorial.ipynb` and `./notebooks/tree_based.ipynb` for training tree-based models or use `sh_scripts/optimize_model.sh` and `sh_scripts/train_model.sh` for building deep learning-based models.