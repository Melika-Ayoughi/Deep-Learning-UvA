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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.embed = nn.Embedding(vocabulary_size, vocabulary_size, _weight=torch.eye(vocabulary_size))
        self.embed.weight.requires_grad = False #don't learn the embeddings
        self.LSTM = nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers, dropout=0)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.to(device)

    def forward(self, x, h_0_c_0=None):
        # assert x.shape == (self.seq_length, self.batch_size)
        x_embedding = self.embed(x)

        output, (hn, cn) = self.LSTM(x_embedding, h_0_c_0) #h0 and c0 are by default initialized to zero
        return self.linear(output), (hn, cn)
