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
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.device = device

        self.params = nn.ParameterDict()
        self.params['W_hx'] = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.params['W_hh'] = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.params['W_ph'] = nn.Parameter(torch.randn(num_hidden, num_classes))
        self.params['b_h'] = nn.Parameter(torch.randn(1, num_hidden))
        self.params['b_p'] = nn.Parameter(torch.randn(1, num_classes))

        self.activation = nn.Tanh()
        self.to(device)

    def forward(self, x):
        assert x.shape == (self.batch_size, self.seq_length, self.input_dim)

        h_t = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        for t in range(self.seq_length):
            # h[t-1]
            h_pt = h_t
            y = x[:, t, :] @ self.params['W_hx'] + h_pt @ self.params['W_hh'] + self.params['b_h']
            h_t = self.activation(y)
        p = h_pt @ self.params['W_ph'] + self.params['b_p']

        return p
