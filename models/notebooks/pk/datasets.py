import torch
import pandas as pd

class PKSmilesDataset(torch.utils.data.Dataset):
    """ Dataset for PK data with SMILES strings as input and AUC as label. """

    def __init__(self, file_path, tokenizer=None, mode='bin_class', 
                 label_col='AUC_bin', smiles_col='mol', threshold=1000):
        self.mode = mode
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.label_col = label_col
        self.threshold = threshold
        self.smiles_col = smiles_col
        
        # Read data file
        self.df = pd.read_csv(self.file_path)
        
        self.y = self.df[self.label_col].values
        self.X = self.df[self.smiles_col].values
                                
        if self.tokenizer:
            if 'MFBERT' in str(type(self.tokenizer)):
                # MFBERT tokenizer
                self.X = tokenizer(
                    [x for x in self.X], 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True
                )
            else:
                # SMILESTransformer tokenizer
                self.X = self.tokenizer.transform(self.X)

    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        if 'MFBERT' in str(type(self.tokenizer)):
            X_inp = {
                'input_ids': self.X['input_ids'][idx],
                'token_type_ids': self.X['token_type_ids'][idx],
                'attention_mask': self.X['attention_mask'][idx]
            }
        else:
            X_inp = self.X[idx]
        return X_inp, self.y[idx]