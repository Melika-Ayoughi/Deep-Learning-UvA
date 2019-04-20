import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormAutograd object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomBatchNormAutograd, self).__init__()

        self.beta = nn.Parameter(torch.zeros(n_neurons))
        self.gamma = nn.Parameter(torch.eye(n_neurons))
        self.eps = eps
        self.n_neurons = n_neurons

    def forward(self, input):
        """
        Compute the batch normalization

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor

        TODO:
          Check for the correctness of the shape of the input tensor.
          Implement batch normalization forward pass as given in the assignment.
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """
        if input.shape[1] != self.n_neurons:
            raise Exception('shape of input does not correspond with the number of neurons')
        mean = input.mean(0)
        variance = input.var(0, unbiased=False)
        input_hat = (input - mean) / torch.sqrt(variance + self.eps)
        out = input_hat @ self.gamma + self.beta
        return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
      my_bn_fct = CustomBatchNormManualFunction()
      normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: batch-normalized tensor

        TODO:
          Implement the forward pass of batch normalization
          Store constant non-tensor objects via ctx.constant=myconstant
          Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
          Intermediate results can be decided to be either recomputed in the backward pass or to be stored
          for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """
        ctx.eps = eps
        N = input.shape[0]
        shifted_mean = input - (1/N) * input.sum(0)
        sq = shifted_mean.pow(2)
        var = 1 / N * sq.sum(0)
        sqrtvar = torch.sqrt(var + eps)
        scaled_var = torch.reciprocal(sqrtvar) # 1/x
        input_hat = shifted_mean * scaled_var
        gammax = gamma * input_hat
        out = gammax + beta

        # store intermediate
        ctx.save_for_backward(input_hat, gamma, shifted_mean, scaled_var, sqrtvar, var)

        return out


    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the batch normalization.

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments

        TODO:
          Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
          Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
          inputs to None. This should be decided dynamically.
        """
        input_hat, gamma, shifted_mean, scaled_var, sqrtvar, var = ctx.saved_tensors
        eps = ctx.eps
        grad_input, grad_gamma, grad_beta = None, None, None
        N, D = grad_output.shape


        if ctx.needs_input_grad[1]:
            grad_gamma = torch.sum(grad_output * input_hat, 0)
        if ctx.needs_input_grad[2]:
            grad_beta = grad_output.sum(0)
        if ctx.needs_input_grad[0]:

            dinput_hat = grad_output * gamma
            # step7
            dscaled_var = torch.sum(dinput_hat * shifted_mean, 0)
            dxmu1 = dinput_hat * scaled_var

            # step6
            dsqrtvar = -1 / (sqrtvar ** 2) * dscaled_var

            # step5
            dvar = 0.5 * 1 / torch.sqrt(var + eps) * dsqrtvar

            # step4
            dsq = torch.ones((N, D), dtype=dvar.dtype) * (1/N) * dvar

            # step3
            dxmu2 = 2 * shifted_mean * dsq

            # step2
            dx1 = (dxmu1 + dxmu2)
            dmu = torch.sum(-1 * (dxmu1 + dxmu2), 0)

            # step1
            dx2 = torch.ones((N, D), dtype=dmu.dtype) * (1/N) * dmu

            # step0
            grad_input = dx1 + dx2

        return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormManualModule, self).__init__()


        self.beta = nn.Parameter(torch.zeros(n_neurons))
        self.gamma = nn.Parameter(torch.ones(n_neurons))
        self.eps = eps
        self.n_neurons = n_neurons


    def forward(self, input):
        """
        Compute the batch normalization via CustomBatchNormManualFunction

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor

        TODO:
          Check for the correctness of the shape of the input tensor.
          Instantiate a CustomBatchNormManualFunction.
          Call it via its .apply() method.
        """

        if input.shape[1] != self.n_neurons:
            raise Exception('shape of input does not correspond with the number of neurons')

        func = CustomBatchNormManualFunction()
        out = func.apply(input, self.gamma, self.beta, self.eps)
        return out
