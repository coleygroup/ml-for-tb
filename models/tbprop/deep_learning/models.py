import torch
import numpy as np
import lightning as L

from MFBERT.Model.model import MFBERT

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from collections import OrderedDict
from transformers import AutoModel

from tbprop.deep_learning.modules import MultiLayerPerceptron, \
    Conv1dBlock, LstmHiddenTensorExtractor, construct_fc_layers


def max_f1_score(y_true, y_score):
    """ 
    Calculates the maximum possible F1 score given true labels and prob scores. 
    
    Parameters
    ----------
    y_true: List[int] or np.array
        True (binary) labels.
    y_score: List[float] or np.array
        Prob scores from the model.
    """
    return max([f1_score(y_true, (np.array(y_score) > t).astype(int)) for t in np.arange(0., 1.01, 0.02)])


class LightningBinaryClassifier(L.LightningModule):
    def __init__(self, config):
        """
        Constructor. Config should at least have the following params: 
        'learning_rate': float

        To construct a lightning classifier, inherit this class and override the 
        following functions:
            __init__()
            forward()
            predict_step() (Optional): use to create embeddings.
            configure_optimizers (Optional): to use an alternate optimizer.
        """
        super().__init__()
        self.pos_weight = config.get('pos_weight', 1.)
        self.learning_rate = config.get('learning_rate', 1e-1)
        self.scale_loss = config.get('scale_loss', False)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([self.pos_weight])
        )
        self.threshold_objective = config.get('threshold_objective', 'f1_score')

        if self.threshold_objective not in ('f1_score', 'youdens_j', 'accuracy'):
            raise ValueError(f"Threshold objective '{self.threshold_objective}' not recognized.")
        elif self.threshold_objective == 'youdens_j':
            raise NotImplementedError(f"Youden's J objective not implemented yet.")
        
        self.objective_mapping = {
            'f1_score': f1_score,
            'accuracy': accuracy_score
        }

        self.objective_fn = self.objective_mapping[self.threshold_objective]

        # Used for accumulating the outputs across validation batches
        self.validation_outputs = []
        self.validation_targets = []

    def forward(self, x):
        """ Forward function connecting torch.nn.Modules. """
        return x

    def training_step(self, batch, batch_idx):
        """ Batch-wise training function. """
        inputs, targets = batch
        outputs = self(inputs).reshape(-1)

        multiplier = 1/inputs.shape[0] if self.scale_loss else 1
        train_loss = multiplier*self.loss_fn(outputs, targets.to(torch.float32))

        self.log("ptl/train_loss", train_loss, on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        """ Calculate batch-wise validation metrics at the end of each epoch. """
        inputs, targets = batch
        outputs = self(inputs).reshape(-1)

        multiplier = 1/inputs.shape[0] if self.scale_loss else 1
        loss = multiplier*self.loss_fn(outputs, targets.to(torch.float32))

        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()
        auroc = roc_auc_score(targets, outputs)
        ap = average_precision_score(targets, outputs)
        max_f1 = max_f1_score(targets, outputs)

        self.validation_outputs.append(outputs)
        self.validation_targets.append(targets)

        self.log("ptl/val_loss", loss)
        self.log("ptl/val_auroc",auroc)
        self.log("ptl/val_ap", ap)
        self.log("ptl/val_f1_score", max_f1)

    def on_validation_epoch_end(self):
        val_outputs = np.concatenate(self.validation_outputs)
        val_targets = np.concatenate(self.validation_targets)

        threshold_scores = []
        for t in np.arange(0., 1.01, 0.02):
            binary_outputs = (val_outputs > t).astype(int)
            threshold_scores.append(self.objective_fn(val_targets, binary_outputs))

        best_threshold = np.argmax(threshold_scores)*0.02
        self.log("ptl/best_threshold", best_threshold)

    def test_step(self, batch, batch_idx):
        """ Calculate batch-wise validation metrics at the end of each epoch. """
        inputs, targets = batch
        outputs = self(inputs).reshape(-1)

        multiplier = 1/inputs.shape[0] if self.scale_loss else 1
        loss = multiplier*self.loss_fn(outputs, targets.to(torch.float32))

        targets = targets.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()
        auroc = roc_auc_score(targets, outputs)
        ap = average_precision_score(targets, outputs)
        max_f1 = max_f1_score(targets, outputs)

        self.log("ptl/test_loss", loss)
        self.log("ptl/test_auroc", auroc)
        self.log("ptl/test_ap", ap)
        self.log("ptl/test_f1_score", max_f1)

    def configure_optimizers(self):
        """ Configure optimizer. """
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


class FullyConnectedBinaryClassifier(LightningBinaryClassifier):
    def __init__(self, config):
        """
        Structure: Embedding -> Flatten -> MLP (x m)
        Hyperparameters:
            seq_len: int
            embed_size: int
            vocab_size: int
            fc_num_layers: int
            dropout_rate: int
            learning_rate: float
        """
        super().__init__(config=config)

        self.seq_len = config.get('seq_len', 512) # Based on ChemBERTa
        self.embed_size = config.get('embed_size', 64)
        self.vocab_size = config.get('vocab_size', 591) # Based on ChemBERTa
        self.dropout_rate = config.get('dropout_rate', 0.25)
        self.fc_num_layers = config.get('fc_num_layers', 5)
        self.fc_end_dim = 1
        self.pad_index = 0

        self.example_input_array = torch.randint(low=0, 
                                                 high=self.vocab_size, 
                                                 size=(1, self.seq_len))

        self.embedding = torch.nn.Embedding(self.vocab_size, 
                                            self.embed_size, 
                                            padding_idx=self.pad_index)
        self.feature_extractor = torch.nn.Sequential(OrderedDict([('flatten', torch.nn.Flatten())]))

        self.fc_start_dim = self.feature_extractor(torch.zeros(1, self.seq_len, self.embed_size)).shape[1]

        self.fc_layer_sizes = construct_fc_layers(self.fc_start_dim, 
                                                  self.fc_num_layers, 
                                                  self.fc_end_dim)

        self.mlp = MultiLayerPerceptron(self.fc_layer_sizes, 
                                        dropout_rate=self.dropout_rate)
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.feature_extractor(x)
        x = self.mlp(x)
        x = x.squeeze()
        return x
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """ Use this as an embedding function. """
        inputs, _ = batch
        x = self.embedding(inputs)
        x = self.feature_extractor(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x.squeeze()
    

class ConvolutionalBinaryClassifier(FullyConnectedBinaryClassifier):
    def __init__(self, config):
        """
        Structure: Embedding -> Conv1dBlock (x 2) -> Flatten -> MLP (x m)
            Conv1dBlock: Conv1d -> BatchNorm -> ReLU -> Dropout -> MaxPool
        Hyperparameters:
            conv1_dim: int
            conv2_dim: int
            kernel_size: int (Optional)
            stride: int (Optional)
            normalize: bool
            seq_len: int
            embed_size: int
            vocab_size: int
            fc_num_layers: int
            dropout_rate: int
            learning_rate: float
        
        Notes:
            1. stride = 1 and kernel size = (3, 3) generally for image-based CNNs
               so we can continue to follow the same for SMILES/text data as well.
            2. The number of convolutional layers is fixed to 2 for now, but this
               can be changed in the future.
        """

        super().__init__(config=config)
        self.conv1_dim = config.get('conv1_dim', 64)
        self.conv2_dim = config.get('conv2_dim', 32)
        self.kernel_size = config.get('kernel_size', 3)
        self.stride = config.get('stride', 1)
        self.normalize = config.get('normalize', True)
        self.conv_layer_sizes = [self.seq_len, self.conv1_dim, self.conv2_dim]
        self.feature_extractor = torch.nn.Sequential()

        for i in range(len(self.conv_layer_sizes)-1):
            self.feature_extractor.add_module(f"conv_block_{i}", 
                                                Conv1dBlock(self.conv_layer_sizes[i], 
                                                            self.conv_layer_sizes[i+1], 
                                                            kernel_size=self.kernel_size, 
                                                            stride=self.stride,
                                                            normalize=self.normalize,
                                                            dropout_rate=self.dropout_rate))
        self.feature_extractor.add_module('flatten', torch.nn.Flatten())

        self.fc_start_dim = self.feature_extractor(torch.zeros(1, self.seq_len, self.embed_size)).shape[1]        
        self.fc_layer_sizes = construct_fc_layers(self.fc_start_dim, self.fc_num_layers, self.fc_end_dim)

        self.mlp = MultiLayerPerceptron(self.fc_layer_sizes, dropout_rate=self.dropout_rate)


class LSTMBinaryClassifier(FullyConnectedBinaryClassifier):
    def __init__(self, config):
        """
        Structure: Embedding -> LSTM (x n) -> ReLU -> Flatten -> MLP (x m)
        Hyperparameters:
            lstm_hidden_size: int
            lstm_num_layers: int
            seq_len: int
            embed_size: int
            vocab_size: int
            fc_num_layers: int
            dropout_rate: int
            learning_rate: float
        """
        super().__init__(config=config)
        self.lstm_hidden_size = config.get('lstm_hidden_size', 64)
        self.lstm_num_layers = config.get('lstm_num_layers', 2)

        self.feature_extractor = torch.nn.Sequential()
        self.feature_extractor.add_module('lstm', torch.nn.LSTM(self.embed_size, 
                                                                self.lstm_hidden_size, 
                                                                num_layers=self.lstm_num_layers, 
                                                                batch_first=True,
                                                                bidirectional=True))
        self.feature_extractor.add_module('lstm_tensor_extractor', LstmHiddenTensorExtractor())
        self.feature_extractor.add_module('lstm_hidden_act', torch.nn.ReLU())
        self.feature_extractor.add_module('flatten', torch.nn.Flatten())

        self.fc_start_dim = self.feature_extractor(torch.zeros(1, self.seq_len, self.embed_size)).shape[1]        
        self.fc_layer_sizes = construct_fc_layers(self.fc_start_dim, self.fc_num_layers, self.fc_end_dim)

        self.mlp = MultiLayerPerceptron(self.fc_layer_sizes, dropout_rate=self.dropout_rate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ChemBERTaBinaryClassifier(LightningBinaryClassifier):
    def __init__(self, config):
        """ 
        Structure: ChemBERTa-v2 -> MLP (x m)
        Hyperparameters:
            fc_num_layers: int
            learning_rate: float
        """
        super().__init__(config=config)
        self.start_size = 384 # fixed for ChemBERTa
        self.end_size = 1
        self.n_layers = config.get('fc_num_layers', 1)

        # self.example_input_array = torch.randint(low=0, 
        #                                          high=591, 
        #                                          size=(1, config.get('seq_len', 128)))

        # Save parameters as attributes
        self.fc_layers = construct_fc_layers(start_size=self.start_size, 
                                             end_size=self.end_size, 
                                             num_layers=self.n_layers)

        # Model components
        self.feature_extractor = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
        self.mlp = MultiLayerPerceptron(fc_layers=self.fc_layers)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """ Forward function connecting torch.nn.Modules. """
        x = self.feature_extractor(**x).last_hidden_state[:, 0, :]
        x = self.mlp(x)
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """ Use this as an embedding function. """
        inputs, _ = batch
        x = self.feature_extractor(**inputs).last_hidden_state[:, 0, :]
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x.squeeze()
        


class MFBERTBinaryClassifier(LightningBinaryClassifier):
    def __init__(self, config):
        """ 
        Structure: MFBERT -> MLP (x m)
        Hyperparameters:
            fc_num_layers: int
            learning_rate: float
        """
        super().__init__(config=config)
        self.start_size = 384 # fixed for ChemBERTa
        self.end_size = 1
        self.n_layers = config['n_layers']

        # Save parameters as attributes
        self.fc_layers = construct_fc_layers(start_size=self.start_size, 
                                             end_size=self.end_size, 
                                             num_layers=self.n_layers)

        # Model components
        self.feature_extractor = MFBERT(weights_dir='MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
        self.mlp = MultiLayerPerceptron(fc_layers=self.fc_layers)

    def forward(self, x):
        """ Forward function connecting torch.nn.Modules. """
        x = self.feature_extractor(**x).last_hidden_state[:, 0, :]
        x = self.mlp(x)
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """ Use this as an embedding function. """
        inputs, _ = batch
        return self.feature_extractor(**inputs).last_hidden_state[:, 0, :]
    

def init_neural_model(model_name: str, config: dict):
    """ 
    Initializes neural network model given the config. 
    
    Parameters
    ----------
    model_name: str
        Name of model. Options ('FCNN', 'CNN', 'LSTM', 'ChemBERTa-v2', 'MFBERT')
    config: Dict[str, Any]
        Config to initialize model with.
    """

    if model_name == 'FCNN':
        model = FullyConnectedBinaryClassifier(config=config)
    elif model_name == 'CNN':
        model = ConvolutionalBinaryClassifier(config=config)
    elif model_name == 'LSTM':
        model = LSTMBinaryClassifier(config=config)
    elif model_name == 'ChemBERTa-v2':
        model = ChemBERTaBinaryClassifier(config=config)
    elif model_name == 'MFBERT':
        model = MFBERTBinaryClassifier(config=config)
    else:
        raise ValueError(f"Model name '{model_name}' not recognized.")

    return model