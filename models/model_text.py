import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions.uniform import Uniform
import pickle
from collections import namedtuple
from itertools import chain, repeat
import numpy as np
from e2cnn import gspaces
from e2cnn import nn

class RCNNTEXT(torch.nn.Module):

    def __init__(self, n_feats = 48, n_features = 512):
        super(RCNNTEXT, self).__init__()
        from e2cnn import nn
        self.r2_act = gspaces.Rot2dOnR2(N=8)

        in_type = nn.FieldType(self.r2_act, n_features*[self.r2_act.trivial_repr])
        self.input_type = in_type

        # convolution 1
        out_type = nn.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma = 0.66, stride=1)
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        from e2cnn import nn
        x = nn.GeometricTensor(input, self.input_type)

        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = x.tensor
        return x 
