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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

from mlp_pytorch import MLP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    predicted_labels = torch.argmax(predictions, dim=1)
    correct_predictions = torch.sum(predicted_labels == targets).sum().item() / len(predicted_labels)
    #######################
    # END OF YOUR CODE    #
    #######################

    return correct_predictions


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """
    print(use_batch_norm)
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Initialize summary writer
    now = datetime.now()
    current_date = now.strftime("%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(log_dir=f"runs/mlp_pytorch/{current_date}")

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=32*32*3,
                n_hidden=hidden_dims,
                n_classes=10,
                use_batch_norm=use_batch_norm).to(device)
    
    # Experimental: Compile the model
    # model = torch.compile(model)
    
    # # Kaiming initialization
    # for name, param in model.named_parameters():
    #     print(name)
    #     if "bias" in name:
    #         torch.nn.init.zeros_(param)
    #     elif name.startswith("layers.0"):
    #         param.data.normal_(0, 1/np.sqrt(param.shape[1]))
    #     else:
    #         torch.nn.init.kaiming_normal_(param)
            
        
    loss_module = nn.CrossEntropyLoss()
    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer

    optimizer = optim.SGD(model.parameters(),
                          lr=lr)
    
    # Best model
    best_model = deepcopy(model)
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in tqdm(range(epochs), desc="Epochs", total=epochs, disable=False, leave=False, colour="blue"):
      # Train Step 
      model.train()
      train_loss = 0

      progress_bar = tqdm(
          enumerate(cifar10_loader['train']),
          desc=f"Training Epoch {epoch + 1}", 
          total=len(cifar10_loader['train']),
          leave=False,
          disable=False,
          colour="green"
      )

      model_plotted = False

      for batch, (X, y) in progress_bar:
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # For the very first batch, we visualize the computation graph in TensorBoard
          if not model_plotted:
              writer.add_graph(model, X)
              model_plotted = True

          # Forward pass
          y_pred = model(X)

          # Compute loss
          loss = loss_module(y_pred, y)
          train_loss += loss.item()

          # Optimizer Zero Grad
          optimizer.zero_grad()

          # Backward pass
          loss.backward()

          # Optimizer Step
          optimizer.step()
          progress_bar.set_postfix(train_loss=train_loss / (batch + 1))
          progress_bar.update()

      train_loss /= len(cifar10_loader['train'])

      writer.add_scalar('Loss/train', train_loss, epoch + 1)
      writer.flush()
      
      # Validation Step
      model.eval()

      val_loss, val_acc = 0, 0


      progress_bar = tqdm(
          enumerate(cifar10_loader['validation']),
          desc=f"Validation Epoch {epoch + 1}",
          total=len(cifar10_loader['validation']),
          leave=False,
          disable=False,
          colour="red"
      )

      with torch.no_grad():
          for batch, (X, y) in progress_bar:
              # Send data to target device
              X, y = X.to(device), y.to(device)

              # Forward pass
              y_pred = model(X)

              # Compute loss
              loss = loss_module(y_pred, y)
              val_loss += loss.item()

              # Compute accuracy
              val_acc += accuracy(y_pred, y)

              progress_bar.set_postfix(val_loss=val_loss / (batch + 1))
              progress_bar.update()

      val_loss /= len(cifar10_loader['validation'])
      val_acc /= len(cifar10_loader['validation'])

      writer.add_scalar('Loss/validation', val_loss, epoch + 1)
      writer.add_scalar('Accuracy/validation', val_acc, epoch + 1)
      writer.flush()

      if val_acc > best_val_acc:
          best_val_acc = val_acc
          best_model = deepcopy(model)
          best_epoch = epoch + 1

    # Test Step
    test_acc = 0

    progress_bar = tqdm(
        enumerate(cifar10_loader['test']),
        desc="Testing",
        total=len(cifar10_loader['test']),
        leave=False,
        disable=False,
        colour="yellow"
    )

    with torch.no_grad():
        for batch, (X, y) in progress_bar:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = best_model(X)

            # Compute accuracy
            test_acc += accuracy(y_pred, y)

            progress_bar.set_postfix(test_acc=test_acc / (batch + 1))
            progress_bar.update()
    
    test_acc /= len(cifar10_loader['test'])

    print(f"Best model found at epoch {best_epoch} with validation accuracy of {best_val_acc * 100:.2f}% and test accuracy of {test_acc * 100:.2f}%")

    # Add hparams to tensorboard and close writer
    writer.add_hparams(
        {
            "lr": lr,
            "use_batch_norm": use_batch_norm,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "data_dir": data_dir
        },
        {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "test_acc": test_acc
        }
    )

    writer.flush()
    writer.close()





    # val_accuracies = ...
    # # TODO: Test best model
    # test_accuracy = ...
    # # TODO: Add any information you might want to save for plotting
    # logging_dict = ...
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, #val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
