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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size

        self.params = nn.ParameterDict()
        self.params['W_hx'] = torch.randn(input_dim, num_hidden)
        self.params['W_hh'] = torch.randn(num_hidden, num_hidden)
        self.params['W_ph'] = torch.randn(num_hidden, num_classes)
        self.params['b_h'] = torch.randn(1, num_hidden)
        self.params['b_p'] = torch.randn(1, num_classes)

    def forward(self, x):

        h_t = nn.zeros(self.batch_size, self.num_hidden)
        for t in range(self.seq_length):
            h_pt = h_t # h[t-1]
            h_t = nn.Tanh(x[t] @ self.params['W_hx'] + h_pt @ self.params['W_hh'] + self.params['b_h'])  # maybe broadcasting i should add axis
        p = h_pt @ self.params['W_ph'] + self.params['b_p']

        return p
