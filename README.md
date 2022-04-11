# ml-for-tb
Project for sharing surrogate models for TB inhibitors, data sharing, and Jupyter notebooks for prototyping.

### Data
the [`data`](./data/) directory can be used for small (a few MB) CSV files for easy access. Larger files can be shared with figshare, gdrive, etc. and downloaded on your local machine or cluster.

#### [`0_raw`](./data/0_raw/)
TODO: raw activity data

#### [`1_docking`](./data/1_docking/)
docking validation data: PDB structures used for redocking, config files, images of docked poses, docking output

Links to data:
* TBD

### Env setup
As the project develops, `environment.yml` and `requirements.txt` can be used to keep python dependencies organized. This could turn into scripts and an installable python package later on.

### Basic, flexible folder structure
```
├── data
│   └── 0_raw            <- The original data. Good for small (a few MB) CSV files and data for debugging. 
│
├── docs                
│   └── data_dictionaries            <- Documentation that briefly explains data. 
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-ncf-initial-data-exploration`.
```

