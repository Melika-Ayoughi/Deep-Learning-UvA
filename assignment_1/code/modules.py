"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """

    self.params = {'weight': np.random.normal(0, 0.0001, (out_features, in_features)), 'bias': np.zeros((out_features, 1))}
    self.grads = {'weight': np.zeros((out_features,in_features)), 'bias': np.zeros((out_features, 1))}

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module

    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    out = (self.params['weight'] @ x.T + self.params['bias']).T
    print(out.shape)
    self.x = x
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    # other ways to do relu np.maximum(0,x) or (abs(x) + x) / 2; but this way is the fastest
    out = x * (x > 0)
    self.grad = (x > 0)
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    b = x.max()
    out = np.exp(x-b) / sum(np.exp(x-b))
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    Implement forward pass of the module. 
    """

    out = - np.sum(np.dot(y, np.log(x)))
    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
