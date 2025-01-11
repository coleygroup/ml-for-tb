import math
import torch
from smiles_transformer.smiles_transformer.pretrain_trfm import TrfmSeq2seq
from MFBERT.Model.model import MFBERT

class MLP(torch.nn.Module):
    """ MLP with 1 Linear layer. """
    
    def __init__(self, seq_len=220, embed_size=64, vocab_size=45, pad_index=0, 
                 stride=1, kernel_size=3, hidden_size=64, hidden_act=True, dropout_rate=0.25):
        super(MLP, self).__init__()
        
        # Embedding layer parameters
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        
        # Conv layer parameters
        self.hidden_size = hidden_size
        
        # Misc
        self.dropout_rate = dropout_rate
        
        # Layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_index)
        self.flatten = lambda x: x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        
        self.fc1 = torch.nn.Linear(self.embed_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        
        self.hidden_act = torch.relu if hidden_act else lambda x: x
        
        if self.dropout_rate:
            self.dropout = torch.nn.Dropout(self.dropout_rate)
            
        self.final_act = torch.sigmoid
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.hidden_act(x)
        if self.dropout_rate:
            x = self.dropout(x)
        x = self.fc2(x)
        x = self.final_act(x)
        
        return x.squeeze()

    def embed(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.hidden_act(x)
        
        return x


class CNN_MLP(torch.nn.Module):
    """ CNN-MLP with 1 Conv layer, 1 Max Pool layer, and 1 Linear layer. """

    def __init__(self, seq_len=220, embed_size=64, vocab_size=45, pad_index=0, 
                 stride=1, kernel_size=3, conv_out_size=64, dropout_rate=0.25):    
        super(CNN_MLP, self).__init__()
        
        # Embedding layer parameters
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        
        # Conv layer parameters
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_out_size = conv_out_size
        
        # Misc
        self.dropout_rate = dropout_rate
        
        # Layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_index)
        
        self.conv = torch.nn.Conv1d(self.seq_len, self.conv_out_size, self.kernel_size, self.stride)
        self.hidden_act = torch.relu
        self.max_pool = torch.nn.MaxPool1d(self.kernel_size, self.stride)
        
        self.flatten = lambda x: x.view(x.shape[0], x.shape[1]*x.shape[2])
        
        self.fc = torch.nn.Linear(self._linear_layer_in_size(), 1)
        if self.dropout_rate:
            self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.final_act = torch.sigmoid
        
    def _linear_layer_in_size(self):
        out_conv_1 = ((self.embed_size - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)
                            
        return out_pool_1*self.conv_out_size
    
    def forward(self, x):
        x = self.embedding(x)
        
        x = self.conv(x)
        x = self.hidden_act(x)
        x = self.max_pool(x)

        x = self.flatten(x)
        
        x = self.fc(x)
        if self.dropout_rate:
            x = self.dropout(x)
        x = self.final_act(x)
        
        return x.squeeze()

    def embed(self, x):
        x = self.embedding(x)
        
        x = self.conv(x)
        x = self.hidden_act(x)
        x = self.max_pool(x)

        x = self.flatten(x)
        
        return x


class LSTM_MLP(torch.nn.Module):
    def __init__(self, seq_len=220, embed_size=64, vocab_size=45, pad_index=0, 
                 stride=1, kernel_size=3, hidden_size=64, hidden_act=False, dropout_rate=0.25):
        """ LSTM-MLP with 1 LSTM layer, and 1 Linear layer. """
        super(LSTM_MLP, self).__init__()
        
        # Embedding layer parameters
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        self.hidden_size = hidden_size
        
        # Misc
        self.dropout_rate = dropout_rate
        
        # Layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_index)
        
        self.lstm = torch.nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.hidden_act = torch.relu if hidden_act else lambda x: x
        self.flatten = lambda x: x.view(x.shape[0], x.shape[1]*x.shape[2])
        
        self.fc = torch.nn.Linear(self.hidden_size*2*self.seq_len, 1)
        if self.dropout_rate:
            self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.final_act = torch.sigmoid
    
    def forward(self, x):
        x = self.embedding(x)
        
        x, _ = self.lstm(x)
        x = self.hidden_act(x)

        x = self.flatten(x.contiguous())
        
        x = self.fc(x)
        if self.dropout_rate:
            x = self.dropout(x)
        x = self.final_act(x)
        
        return x.squeeze()
    
    def embed(self, x):
        x = self.embedding(x)
        
        x, _ = self.lstm(x)
        x = self.hidden_act(x)
        
        x = self.flatten(x.contiguous())
        
        return x


class SMILESTransformer(torch.nn.Module):
    """ SMILES Transformer with 1 linear layer. """

    def __init__(self, ckpt_path, vocab_size, 
                 freeze_transformer=True, verbose=True):
        super().__init__()
        
        self.ckpt_path = ckpt_path
        self.freeze_transformer = freeze_transformer
        self.verbose = verbose
        
        # Load transformer and encoder
        self.transformer = TrfmSeq2seq(vocab_size, 256, vocab_size, 4)
        model_weights = torch.load(self.ckpt_path, map_location=torch.device('cpu'))
        self.transformer.load_state_dict(model_weights)
        
        if self.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # FC layer to finetune
        self.fc = torch.nn.Linear(1024, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = torch.tensor(self.transformer.encode(torch.t(x)))
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

    def embed(self, x):
        x = torch.tensor(self.transformer.encode(torch.t(x)))

        return x


class MFBERTTransformer(torch.nn.Module):
    """ MFBERT with 1 linear layer. """

    def __init__(self, ckpt_path, vocab_size=None, 
                 freeze_transformer=True, verbose=True):
        super().__init__()
        
        self.ckpt_path = ckpt_path
        self.freeze_transformer = freeze_transformer
        self.verbose = verbose
        self.vocab_size = vocab_size
        
        # Load transformer and encoder
        self.transformer = MFBERT(weights_dir='MFBERT/Model/pre-trained', 
                                  return_attention=False, 
                                  inference_method='mean')

        if self.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # FC layer to finetune
        self.fc = torch.nn.Linear(768, 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        
        return x

    def embed(self, x):
        x = self.transformer(x)

        return x
