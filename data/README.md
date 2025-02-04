# About the Data

This folder contains the following data:

- [0_raw.tar.xz](0_raw.tar.xz) for data used for training machine leanring models. After uncompressing with
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
      └── taacf_val.cs
  ```

- [1_redocking](1_redocking) for redocking related configurations and results.
- [2_molpal](2_molpal) for MolPAL related configurations and results.
