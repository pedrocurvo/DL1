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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict
import numpy as np


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.

        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """
        super(MLP, self).__init__()
        self.layers = []
        layer_sizes = [n_inputs] + n_hidden + [n_classes]
        for i in range(len(layer_sizes) - 1):
            linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])

            # Kaiming initialization
            if not self.layers:
                linear_layer.weight.data.normal_(0, 1/np.sqrt(linear_layer.weight.shape[1])) # Just because I'm not sure if nn.init works for the first layer
            else:
                nn.init.kaiming_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)

            self.layers.append(linear_layer)

            # Batch normalization
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            
            # ELU
            self.layers.append(nn.ELU())
        
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        x = x.view(x.size(0), -1) # Flatten the input because the input is an image
        for layer in self.layers:
            x = layer(x)

        return x

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device

