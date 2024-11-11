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
from skimage.transform import resize
import torchvision.models as models
from utils.train_utils import *
from utils.model_utils import *

class C8NonSteerableCNN(torch.nn.Module):
    
    def __init__(self, n_feats = 48):
        
        super(C8NonSteerableCNN, self).__init__()
        
        
       
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,
                     n_feats*8,
                     kernel_size=5,
                     padding = 1,
                     bias=False),        
            torch.nn.BatchNorm2d(n_feats*8),
            torch.nn.ReLU(inplace = True)
        
        )
        
       
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(n_feats*8,
                     n_feats*8,
                     kernel_size=5,
                     padding = 2,
                     bias=False),
       
            torch.nn.BatchNorm2d(n_feats*8),
            torch.nn.ReLU(inplace = True)
 
        )
        self.pool1 = torch.nn.AvgPool2d(kernel_size  = 2, stride = 2)

        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(n_feats*8,
                     n_feats*8,
                     kernel_size=3,
                     padding = 1,
                     bias=False),
            torch.nn.BatchNorm2d(n_feats*8),
            torch.nn.ReLU(inplace = True)
           
        )
        
        
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(n_feats*8,
                     n_feats*8,
                     kernel_size=3,
                     padding = 1,
                     bias=False),
 
            torch.nn.BatchNorm2d(n_feats*8),
            torch.nn.ReLU(inplace = True)
          
        )
        self.pool2 = torch.nn.AvgPool2d(kernel_size  = 2, stride = 2)
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(n_feats*8,
                     n_feats*8,
                     kernel_size=3,
                     padding = 1,
                     bias=False),
 
            torch.nn.BatchNorm2d(n_feats*8),
            torch.nn.ReLU(inplace = True)
          
        )
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(n_feats*8,
                     n_feats*8,
                     kernel_size=3,
                     padding = 1,
                     bias=False),
 
            torch.nn.BatchNorm2d(n_feats*8),
            torch.nn.ReLU(inplace = True)
          
        )
 
        self.pool3 = torch.nn.AvgPool2d(kernel_size  = 2, stride = 2)
        self.block7 = torch.nn.Sequential(
            torch.nn.Conv2d(n_feats*8,
                     n_feats*8,
                     kernel_size=3,
                     padding = 1,
                     bias=False),
 
            torch.nn.BatchNorm2d(n_feats*8),
            torch.nn.ReLU(inplace = True)
          
        )
        self.block8 = torch.nn.Sequential(
            torch.nn.Conv2d(n_feats*8,
                     n_feats*8,
                     kernel_size=3,
                     padding = 1,
                     bias=False),
 
            torch.nn.BatchNorm2d(n_feats*8),
            torch.nn.ReLU(inplace = True)
          
        )
        self.pool4 = torch.nn.AvgPool2d(kernel_size  = 2, stride = 1)
        
    def forward(self, input: torch.Tensor):
        print("input shifts :: ")
        print("input :: ", input.shape)
        x = self.block1(input)
        print("block1 :: ", x.shape)
        x = self.block2(x)
        print("block2 :: ", x.shape)
        x = self.pool1(x)   
        print("pool1 :: ", x.shape) 
        x = self.block3(x)
        print("block3 :: ", x.shape)
        x = self.block4(x)
        print("block4 :: ", x.shape)
        x = self.pool2(x)
        print("pool2 :: ", x.shape)
        x = self.block5(x)
        print("block5 :: ", x.shape)
        x = self.block6(x)
        print("block6 :: ", x.shape)
        x = self.pool3(x)
        print("pool3 :: ", x.shape)
        x = self.block7(x)
        print("block7 :: ", x.shape)
        x = self.block8(x)
        print("block8 :: ", x.shape)
        x = self.pool4(x)
        print("pool4 :: ", x.shape)
        print(1/0)
        return x    
    
        
class C8SteerableCNN(torch.nn.Module):
    
    def __init__(self, n_feats = 48):
        
        super(C8SteerableCNN, self).__init__()
        from e2cnn import nn
        # the model is equivariant under rotations by 45 (2pi/8) degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
           # nn.MaskModule(in_type, 224, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 32 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma = 0.66, stride=2)
        )
        
        # convolution 3
        in_type = self.block2.out_type
        out_type = nn.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.PointwiseAvgPoolAntialiased(out_type, sigma = 0.66, stride=2)
        
        in_type = self.block4.out_type
        out_type = nn.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr])
        
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma = 0.66, stride=2)
        
        in_type = self.block6.out_type
        out_type = nn.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr])
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.block8 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool4 = nn.PointwiseAvgPoolAntialiased(out_type, sigma = 0.66, stride=1)
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        from e2cnn import nn
        x = nn.GeometricTensor(input, self.input_type)
            
        x = self.block1(x)   
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        
        x = self.block7(x)
        x = self.block8(x)
        x = self.pool4(x)
        
        x = x.tensor
        return x  