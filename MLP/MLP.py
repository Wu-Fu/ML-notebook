import numpy as np
import torch
import torch.nn as nn


class MLP(nn.module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(MLP, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.hidden = nn.Linear(input_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, output_channels)

    def forward(self, X):
        hidden = nn.Softmax(self.hidden(X))
        out = nn.Softmax(self.out(hidden))
        return out

