# About the Data

This folder contains the following data:

- [0_raw.tar.xz](0_raw/0_raw.tar.xz) located in [0_raw](o_raw) for data used for training machine
  leanring models. After uncompressing with
  `tar -xf 0_raw.tar.xz`, the data will be stored in the `0_raw` folder with the following structure:

  ```bash
  0_raw
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

- [1_redocking](1_redocking) docking validation data: PDB structures used for redocking,
  config files, images of docked poses, docking outputs.
- [2_molpal](2_molpal) for MolPAL related configurations and results.
