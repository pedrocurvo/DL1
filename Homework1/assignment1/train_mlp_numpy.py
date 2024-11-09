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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    """

    predicted_labels = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_labels == targets)
    accuracy = correct_predictions / len(targets)

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    all_predictions = []
    all_targets = []

    for x, y in tqdm(data_loader, desc='Eval', leave=False, colour="red"):
        x = x.reshape(x.shape[0], -1)
        predictions = model.forward(x)
        all_predictions.append(predictions)
        all_targets.append(y)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_accuracy = accuracy(all_predictions, all_targets)

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Create a writer
    # Get Current Date and Time to name the model
    now = datetime.now()
    current_date = now.strftime("%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(log_dir=f"runs/mlp_numpy/{current_date}")


    model = MLP(n_inputs=3072, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()

    val_accuracies = []
    best_model = None
    best_val_accuracy = 0

    # Loop through training and testing steps for a number of epochs
    progress_bar_epochs = tqdm(
        range(epochs),
        desc="Epochs",
        total=epochs,
        disable=False,
        leave=False,
        colour="blue"
    )

    for epoch in progress_bar_epochs:
        progress_bar_epochs.set_description(f"Epoch {epoch+1}")
        epoch_loss = 0
        model.clear_cache()

        progress_bar = tqdm(cifar10_loader['train'], desc=f'Train', leave=False, colour="green")

        for x, y in progress_bar:
            # Forward pass
            x = x.reshape(x.shape[0], -1)

            logits = model.forward(x)
            loss = loss_module.forward(logits, y)
            epoch_loss += loss

            progress_bar.set_postfix(loss=loss)
            progress_bar.update()

            # Backward pass
            dout = loss_module.backward(logits, y)
            model.backward(dout)

            # Update weights
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    for param in layer.params:
                        layer.params[param] -= lr * layer.grads[param]

        # Evaluate model on validation set
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_accuracy)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

        # Add values to tensorboard
        writer.add_scalar('Loss/train', epoch_loss / len(cifar10_loader['train']), epoch + 1)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch + 1)

        progress_bar_epochs.update()

    
    # Evaluate best model on test set
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f'Best model validation accuracy: {best_val_accuracy:.4f} - Test accuracy: {test_accuracy:.4f}')

    # Add test accuracy to tensorboard
    writer.add_hparams(
        {
          'test_accuracy': test_accuracy,
        },
        {
          'test_accuracy': test_accuracy,
        }
    )

    logging_dict = {
        'hidden_dims': hidden_dims,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'seed': seed,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy
    }

    writer.close()
    

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

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
