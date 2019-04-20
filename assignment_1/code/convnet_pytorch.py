"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem



        Implement initialization of the network.
        """
        super().__init__()

        self.layers = nn.Sequential(
            self.conv_layer(3, 64, 3, 1, 1),
            self.pooling_layer(3, 2, 1),
            self.conv_layer(64, 128, 3, 1, 1),
            self.pooling_layer(3, 2, 1),
            self.conv_layer(128, 256, 3, 1, 1),
            self.conv_layer(256, 256, 3, 1, 1),
            self.pooling_layer(3, 2, 1),
            self.conv_layer(256, 512, 3, 1, 1),
            self.conv_layer(512, 512, 3, 1, 1),
            self.pooling_layer(3, 2, 1),
            self.conv_layer(512, 512, 3, 1, 1),
            self.conv_layer(512, 512, 3, 1, 1),
            self.pooling_layer(3, 2, 1),
            nn.AvgPool2d(1, 1, 0),

        )
        self.linear = nn.Linear(512, 10, True)

    def conv_layer(self, in_channels, out_channels, k, s, p):
        layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,k , s, p),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
        )
        return layer

    def pooling_layer(self, k, s, p):
        layer = nn.Sequential(
            nn.MaxPool2d(k, s, p)
        )
        return layer


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
        output = self.layers(x)
        # reduce the axis that are 1*1 to have a vector as an input to linear layer
        output = output.view(output.shape[0], -1)
        return self.linear(output)