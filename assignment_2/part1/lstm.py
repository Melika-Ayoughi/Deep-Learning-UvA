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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.device = device

        self.params = nn.ParameterDict()
        self.params['W_gx'] = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.params['W_gh'] = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.params['b_g'] = nn. Parameter(torch.randn(1, num_hidden))

        self.params['W_ix'] = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.params['W_ih'] = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.params['b_i'] = nn.Parameter(torch.randn(1, num_hidden))

        self.params['W_fx'] = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.params['W_fh'] = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.params['b_f'] = nn.Parameter(torch.randn(1, num_hidden))

        self.params['W_ox'] = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.params['W_oh'] = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.params['b_o'] = nn.Parameter(torch.randn(1, num_hidden))

        self.params['W_ph'] = nn.Parameter(torch.randn(num_hidden, num_classes))
        self.params['b_p'] = nn.Parameter(torch.randn(1, num_classes))

        self.tanh_activation = nn.Tanh()
        self.sig_activation = nn.Sigmoid()

        self.to(device)

    def forward(self, x):
        assert x.shape == (self.batch_size, self.seq_length, self.input_dim)

        h_t = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        c_t = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        for t in range(self.seq_length):
            h_pt = h_t
            c_pt = c_t
            g = self.tanh_activation(x[:, t, :] @ self.params['W_gx'] + h_pt @ self.params['W_gh'] + self.params['b_g'])
            i = self.sig_activation(x[:, t, :] @ self.params['W_ix'] + h_pt @ self.params['W_ih'] + self.params['b_i'])
            f = self.sig_activation(x[:, t, :] @ self.params['W_fx'] + h_pt @ self.params['W_fh'] + self.params['b_f'])
            o = self.sig_activation(x[:, t, :] @ self.params['W_ox'] + h_pt @ self.params['W_oh'] + self.params['b_o'])

            c_t = g * i + c_pt * f
            h_t = self.tanh_activation(c_t) * o

        p = h_t @ self.params['W_ph'] + self.params['b_p']
        return p

