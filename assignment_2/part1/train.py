################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def accuracy_(predictions, targets):

    predicted_labels = predictions.argmax(dim=1)
    target_labels = targets #.long()

    return ((predicted_labels == target_labels).float()).mean()

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if (config.model_type == 'RNN'):
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, config.device)
    # else:
    #     model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, config.device)


    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate) #, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)

    accuracies = []
    losses = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = batch_inputs[..., None] # need to add this because input is a number
        batch_inputs.to(device)
        batch_targets.to(device)

        batch_predictions = model.forward(batch_inputs)
        loss = criterion(batch_predictions, batch_targets)
        losses.append(loss.item())
        # model.zero_grad() should we do this??
        loss.backward()


        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm) #prevents maximum gradient problem

        optimizer.step() #before or after clip_grad_norm?

        accuracy = accuracy_(batch_predictions, batch_targets)
        accuracies.append(accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    return losses, accuracies


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)