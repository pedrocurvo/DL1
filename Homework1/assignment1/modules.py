################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        # Initialize weights with Kaiming initialization
        if input_layer:
            self.params['weight'] = np.random.randn(in_features, out_features) * np.sqrt(2/in_features)
        else:
            self.params['weight'] = np.random.randn(in_features, out_features) * np.sqrt(2/in_features)

        # Initialize biases with zeros
        self.params['bias'] = np.zeros((1, out_features))
        
        # Zero-initialize gradients
        self.grads['weight'] = np.zeros_like(self.params['weight'])
        self.grads['bias'] = np.zeros_like(self.params['bias'])


    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        self.x = x
        out = np.dot(x, self.params['weight']) + self.params['bias']

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        dx = np.dot(dout, self.params['weight'].T)
        self.grads['weight'] = np.dot(self.x.T, dout)
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.x = None


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.x = x
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        """

        dx = np.where(self.x > 0, dout, dout * self.alpha * np.exp(self.x))
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        """
        self.x = None


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

        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.x = x
        # Max trick
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        self.output = out
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        """
        s = self.output
        
        # Reshape s and dout for batch matrix multiplication
        s_reshaped = s[:, :, np.newaxis] # (batch_size, num_classes, 1)
        dout_reshaped = dout[:, np.newaxis, :] # (batch_size, 1, num_classes)
        
        # Compute outer product for each item in batch
        outer_prod = s_reshaped @ dout_reshaped # (batch_size, num_classes, num_classes)
        
        # Create diagonal matrices for each item in batch
        diag = np.zeros_like(outer_prod)
        indices = np.arange(s.shape[1])
        diag[:, indices, indices] = s
        
        # Compute the final gradient
        # dx = s * (diag - s^T s) * dout
        dx = s * (dout - np.sum(dout * s, axis=1, keepdims=True))

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        """
        self.x = None
        self.output = None


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

        """

        one_hot_y = np.zeros((y.size, x.shape[-1]))
        one_hot_y[np.arange(y.size), y] = 1
        y = one_hot_y
            
        self.y = y
        self.p = x
        out = -np.sum(y * np.log(x + 1e-12)) / x.shape[0]

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        """
        one_hot_y = np.zeros((y.size, x.shape[-1]))
        one_hot_y[np.arange(y.size), y] = 1
        y = one_hot_y
        dx = -y / (x + 1e-12) / x.shape[0]

        return dx