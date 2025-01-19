import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class SmilesDataset(Dataset):
    """Dataset for PK data with SMILES strings as input and AUC as label."""

    def __init__(
        self,
        file_path,
        tokenizer,
        model_name,
        target_col="AUC_bin",
        smiles_col="mol",
        tokenizer_max_length=512,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.tokenizer_max_length = tokenizer_max_length

        self.recognized_model_names = (
            "FCNN",
            "CNN",
            "LSTM",
            "SmilesTransformer",
            "MFBERT",
            "ChemBERTa-v2",
        )

        if self.model_name not in self.recognized_model_names:
            raise ValueError(
                f"""Model name {self.model_name} not recognized. 
                Recognized model names are: {self.recognized_model_names}."""
            )

        # Read data file
        self.df = pd.read_csv(self.file_path)

        self.y = torch.from_numpy(self.df[self.target_col].values)
        self.X = self.df[self.smiles_col].tolist()

        if self.model_name in ("FCNN", "CNN", "LSTM", "SmilesTransformer", "ChemBERTa-v2"):
            self.encodings = tokenizer(
                self.X,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_max_length,
                return_tensors="np",
            )
        elif self.model_name == "MFBERT":
            self.encodings = tokenizer(self.X, return_tensors="np")

    def encoding_shape(self):
        return self.encodings["input_ids"].shape

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if self.model_name in ("MFBERT", "ChemBERTa-v2"):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        else:
            item = torch.tensor(self.encodings["input_ids"][idx])
        return item, self.y[idx]


class SmilesDataModule:
    def __init__(
        self,
        dataset: str,
        data_dir: str,
        tokenizer,
        model_name: str,
        target_col: str,
        smiles_col: str,
        train_batch_size: int = 512,
        infer_batch_size: int = 512,
        num_workers: int = 7,
        tokenizer_max_length: int = 512,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.target_col = target_col
        self.smiles_col = smiles_col
        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.num_workers = num_workers
        self.tokenizer_max_length = tokenizer_max_length

        self.dataset_prefix = f"{self.data_dir}/{self.dataset}/{self.dataset}"

        self.trn_set = SmilesDataset(
            f"{self.dataset_prefix}_trn.csv",
            self.tokenizer,
            model_name=self.model_name,
            target_col=self.target_col,
            smiles_col=self.smiles_col,
            tokenizer_max_length=self.tokenizer_max_length,
        )
        self.val_set = SmilesDataset(
            f"{self.dataset_prefix}_val.csv",
            self.tokenizer,
            model_name=self.model_name,
            target_col=self.target_col,
            smiles_col=self.smiles_col,
            tokenizer_max_length=self.tokenizer_max_length,
        )
        self.tst_set = SmilesDataset(
            f"{self.dataset_prefix}_tst.csv",
            self.tokenizer,
            model_name=self.model_name,
            target_col=self.target_col,
            smiles_col=self.smiles_col,
            tokenizer_max_length=self.tokenizer_max_length,
        )

        self.trn_val_set = ConcatDataset([self.trn_set, self.val_set])
        self.full_set = ConcatDataset([self.trn_set, self.val_set, self.tst_set])

        # Training dataloaders
        self.trn_dataloader = DataLoader(
            self.trn_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

        self.trn_val_dataloader = DataLoader(
            self.trn_val_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

        # Inference dataloaders
        self.val_dataloader = DataLoader(
            self.val_set,
            batch_size=self.infer_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )

        self.tst_dataloader = DataLoader(
            self.tst_set,
            batch_size=self.infer_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )

        self.full_dataloader = DataLoader(
            self.full_set,
            batch_size=self.infer_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )

    def encoding_shape(self, split: str):
        if split == "train":
            return self.trn_set.encoding_shape()
        elif split == "validation":
            return self.val_set.encoding_shape()
        elif split == "test":
            return self.tst_set.encoding_shape()

        raise ValueError(f"split '{split}' not recognized")
