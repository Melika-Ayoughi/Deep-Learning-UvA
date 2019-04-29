# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################
def accuracy_(predictions, targets):

    predicted_labels = predictions.argmax(dim=1)
    target_labels = targets #.long()

    return ((predicted_labels == target_labels).float()).mean()

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length) # should we do +1??
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                                config.lstm_num_hidden, config.lstm_num_layers, device)


    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    losses = []
    accuracies = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # batch_inputs = batch_inputs  # need to add this because input is a number
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        batch_predictions = model.forward(batch_inputs)
        loss = criterion(batch_predictions, batch_targets)
        losses.append(loss.item())
        model.zero_grad()  # should we do this??
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)  # prevents maximum gradient problem

        optimizer.step()  # before or after clip_grad_norm?

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.sample_every:
            # Generate some sentences by sampling from the model
            pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    # save only the model parameters
    torch.save(model.state_dict(), config.save_file)
    # revive the model
    # model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size(),
    #                                 config.lstm_num_hidden, config.lstm_num_layers, device)
    # model.load_state_dict(torch.load(config.save_file))

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default="./tiny-shakespeare.txt", help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--save_file', type=str, default="./outputs/saved_model.txt", help="Path to a .txt file to save the model on")
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
