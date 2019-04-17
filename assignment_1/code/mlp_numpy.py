"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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

    self.layers = []
    n_previous = n_inputs
    for n_current in n_hidden:
      self.layers.append(LinearModule(n_previous,n_current))
      self.layers.append(ReLUModule())
      n_previous = n_current

    #last layer has no relu module and linear module directly connects to softmax
    self.layers.append(LinearModule(n_previous, n_classes))
    self.layers.append(SoftMaxModule())

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
      x = module.forward(x)
    return x

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss

    Implement backward pass of the network.
    """

    for module in reversed(self.layers):
      dout = module.backward(dout)
    return
