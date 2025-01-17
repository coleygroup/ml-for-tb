import torch
import numpy as np
from typing import List

class Conv1dBlock(torch.nn.Sequential):
    """
    A Conv1d block with Conv1d, BatchNorm1d, ReLU, Dropout and MaxPool1d layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 hidden_act=torch.nn.ReLU(), normalize=True, dropout_rate=0.25):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Kernel size for the Conv1d layer. Default: 3.
        stride : int
            Stride for the Conv1d layer. Default: 1.
        padding : int
            Padding for the Conv1d layer. Default: 0.
        hidden_act : torch.nn.Module
            Hidden activation function. Default: torch.nn.ReLU().
        normalize : bool
            Whether to use BatchNorm1d. Default: True.
        dropout_rate : float
            Dropout rate for Dropout layer. Default: 0.25.
        """
        super(Conv1dBlock, self).__init__()

        self.add_module(f"conv", torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
        if normalize:
            self.add_module(f"bn", torch.nn.BatchNorm1d(out_channels))
        self.add_module(f"act", hidden_act)
        if dropout_rate > 0:
            self.add_module(f"dropout", torch.nn.Dropout(dropout_rate))
        self.add_module(f"maxpool", torch.nn.MaxPool1d(kernel_size, stride))


class MultiLayerPerceptron(torch.nn.Sequential):
    """ 
    MLP with repeating FC-ReLU-Dropout layers.
    """
    
    def __init__(self, fc_layers, dropout_rate=0.25, final_act=None):
        """
        Parameters
        ----------
        fc_layers : list
            List of FC layer sizes. Start with the input size and end with no. of classes 
            for classification and no. of labels for regression.
            E.g. [128, 64, 32, 8, 1] for a 5-layer binary classification MLP.
        dropout_rate : float
            Dropout rate for dropout layers. Default: 0.
        task : str
            Task type. Either 'bin_class', 'multi_class' or 'regression'. 
            Default: 'bin_class'.
        """

        super(MultiLayerPerceptron, self).__init__()

        self.fc_layers = fc_layers
        self.dropout_rate = dropout_rate
        self.final_act = final_act

        if final_act and final_act not in ('sigmoid', 'softmax'):
            raise ValueError(f"final_act should either be None or 'sigmoid' or 'softmax', not '{final_act}'")

        layers = []
        for i in range(len(fc_layers) - 1):
            layers.append((f"fc_{i}", 
                        torch.nn.Linear(fc_layers[i], 
                                        fc_layers[i+1])))
            if i != len(fc_layers) - 2:
                layers.append((f"relu_{i}", torch.nn.ReLU()))
                if dropout_rate > 0:
                    layers.append((f"dropout_{i}", torch.nn.Dropout(dropout_rate)))

        if final_act:
            if final_act == 'sigmoid':
                layers.append((f"sigmoid", torch.nn.Sigmoid()))
            elif final_act == 'softmax':
                layers.append((f"softmax", torch.nn.Softmax(dim=1)))

        for name, module in layers:
            self.add_module(name, module)


class LstmHiddenTensorExtractor(torch.nn.Module):
    """
    LSTM hidden tensor extractor.
    """
    def forward(self, x):
        """
        Parameters
        ----------
        x : tuple
            A tuple of (output, (h_n, c_n)) from LSTM.

        Returns
        -------
        torch.Tensor
            The hidden tensor (h_n).
        """
        return x[0][:, -1, :]


def construct_fc_layers(start_size: int, num_layers: int, end_size: int = 1) -> List[int]:
    """
    Constructs the dimensions of fully connected layers of the MultiLayerPerceptron
    class. Given a start_size and end_size, start with the highest power of 2 lower
    than or equal to start_size and cascade by powers of 2 until either num_layers
    or end_size is reached.

    Parameters
    ----------
    start_size: int
        Start dimension of the MLP
    end_size: int
        End dimension of the MLP
    num_layers: int
        Number of layers to have in the MLP
    """
    assert start_size > end_size and num_layers > 0, \
        "Constraints on start_size, end_size, and num_layers have failed."

    start_pow_2, end_pow_2 = int(np.log2(start_size)), int(np.log2(end_size))
    max_layers = start_pow_2 - end_pow_2 - 1

    if max_layers < num_layers:
        raise ValueError(
            f"num_layers should be <= max_layers. max_layers = {max_layers}."
        )

    layers = list(range(start_pow_2-1, start_pow_2 - num_layers, -1))

    return [start_size] + [2**x for x in layers] + [end_size]
