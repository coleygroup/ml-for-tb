# tb-pk-mic
PK and MIC models for screening candidates for TB.

### Setup

Create a conda environment with the required dependencies:
```bash
conda env create --file environment.yml
conda activate tbprop
```

### Inference with the PK Models

You can use the inference script in the following way:

1. Calculate the [MOE 2D descriptors](https://www.chemcomp.com/Products.htm) (see documentation of descriptors [here](https://cadaster.eu/sites/cadaster.eu/files/challenge/descr.htm).) for the compounds in the inference set. A sample of how the output of this should look is in `vedang/pk/data/sample_inference_set.csv`.
2. Run the inference script:
```bash
cd vedang/pk
python inference.py --infer_path /path/to/inference/set.csv --output_path /path/to/output.csv --verbose
```
Please also check other arguments that can be passed to the inference script by running `python inference.py --help`. These include the choice of model (`RandomForestClassifier/optuna/P2` is recommended), and the name of the SMILES column in the inference set.