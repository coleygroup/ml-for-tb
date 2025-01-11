import torch
import pickle

from smiles_transformer.smiles_transformer.build_vocab import WordVocab
from smiles_transformer.smiles_transformer.utils import split

class SmilesTokenizer:
    def __init__(self, vocab_path, pad_index=0, unk_index=1, 
                 eos_index=2, sos_index=3, mask_index=4, seq_len=220):
        self.vocab_path = vocab_path
        self.pad_index = pad_index   # Padding
        self.unk_index = unk_index   # Unknown
        self.eos_index = eos_index   # End-of-sequence
        self.sos_index = sos_index   # Start-of-sequence
        self.mask_index = mask_index # Not used in finetuning/inference
        self.seq_len = seq_len
        self.max_len = seq_len - 2   # To accommodate [SOS] and [EOS]
        
        # Load vocab
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            
        self.vocab_size = len(self.vocab)
        
    def _preprocess_mol(self, mol):
        mol = list(mol)
        if len(mol) > self.max_len:
            mol = mol[:self.max_len // 2] + mol[-self.max_len // 2:]
        ids = [self.vocab.stoi.get(token, self.unk_index) for token in mol]
        ids = [self.sos_index] + ids + [self.eos_index]
        padding = [self.pad_index]*(self.seq_len - len(ids))
        ids += padding
        return ids
        
    def transform(self, x):
        return torch.tensor([self._preprocess_mol(mol) for mol in x])