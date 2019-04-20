"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP


        Implement initialization of the network.
        """
        super().__init__()

        # we need to give torch the layers
        self.layers = nn.ModuleList()
        n_previous = n_inputs
        for n_current in n_hidden:
            self.layers.append(nn.Linear(n_previous,n_current))
            self.layers.append(nn.ReLU(True))
            n_previous = n_current

        #last layer has no relu module and linear module directly connects to softmax
        self.layers.append(nn.Linear(n_previous, n_classes))

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        Implement forward pass of the network.
        """

        for module in self.layers:
            x = module(x)

        return x
