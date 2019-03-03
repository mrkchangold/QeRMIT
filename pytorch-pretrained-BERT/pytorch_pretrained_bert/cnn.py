import sys
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, e_T = 50, k = 5):
        # e_T is transformer size
        #
        super(CNN, self).__init__()
        self.e_T = e_T
        self.qconv = None
        self.k = k
        self.convLayer = nn.Conv1d(in_channels = e_T, out_channels = e_T, kernel_size = k, stride=1, padding=0, bias=True)

    def forward(self, qreshaped: torch.Tensor):
        self.qconv = self.convLayer(qreshaped)
        qconv_out = nn.MaxPool1d(kernel_size = self.qconv.size()[-1] - self.k + 1)(nn.ReLU()(self.qconv)) # potentially ask question
        return qconv_out

### END YOUR CODE
