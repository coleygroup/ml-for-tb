# ml-for-tb

Project for sharing surrogate models for TB inhibitors, data sharing, and Jupyter notebooks for prototyping.

## Data

The [`data`](/data) directory contains data for training surrogate models, configurations and
results for redocking and MolPAL
based virtual screening. More details are listed in [/data/README.md](/data/README.md).

- [0_raw/0_raw.tar.xz](0_raw/0_raw.tar.xz) for data used for training machine leanring models. After uncompressing with
  `tar -xf 0_raw.tar.xz`, the data will be stored in the `0_raw` folder with the following structure:

  ```bash
  .
  ├── mlsmr
  │   ├── mlsmr_trn.csv
  │   ├── mlsmr_tst.csv
  │   └── mlsmr_val.csv
  ├── pk
  │   ├── pk_trn.csv
  │   ├── pk_tst.csv
  │   └── pk_val.csv
  └── taacf
      ├── taacf_trn.csv
      ├── taacf_tst.csv
      └── taacf_val.csv
  ```

- [1_redocking](1_redocking) docking validation data: PDB structures used for redocking, config files, images of docked poses, docking outputs.
- [2_molpal](2_molpal) for MolPAL related configurations and results.

## Environment Setup

As the project develops, `environment.yml` and `requirements.txt` can be used to keep python dependencies organized. This could turn into scripts and an installable python package later on.

## Models

The [`models`](/models) directory contains trained surrogate models for TB inhibitors. More details
are listed in [/models/README.md](/models/README.md).

## Notebooks

The [`notebooks`](/notebooks) directory contains Jupyter notebooks for pre-processing and
post-processing molecular library for molecular docking and virtual screening.
