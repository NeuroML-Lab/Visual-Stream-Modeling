from torch import nn
import torch
import numpy as np
import math
from torch.nn import functional as F

def positive(weight):
    """
    Enforces tensor to be positive. Changes the tensor in place. Produces no gradient.
    Args:
        weight: tensor to be positive
    """
    
    weight.data *= weight.data.ge(0).float()
    
    
class Encoder(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout
       

    def forward(self, x, data_key=None, detach_core=False, fake_relu = False, **kwargs):
        x = self.core(x)
        if detach_core:
            x = x.detach()
        if "sample" in kwargs:
            x = self.readout(x,  sample=kwargs["sample"])
        else:
            x = self.readout(x)
        return x 
    

class Encoder_semantic(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout
       

    def forward(self, x, img, data_key=None, detach_core=False, fake_relu = False, **kwargs):
        feats = self.core(x)
        if detach_core:
            feats = feats.detach()
        
        x = self.readout(feats, img)
        return x 
    
class Encoder_semantic_text_image(nn.Module):
    def __init__(self, core_img, core_txt, readout):
        super().__init__()
        self.core_img = core_img
        self.core_txt = core_txt
        self.readout = readout
       

    def forward(self, x, img, data_key=None, detach_core=False, fake_relu = False, **kwargs):
        feats_img = self.core_img(img)
        feats_txt = self.core_txt(x)
        x = self.readout(feats_img, feats_txt, img, x)
        return x 
    
class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=1.0):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
    
    def forward(self, x):
        return self.linear(x)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.).cuda()
        for param in self.parameters():
            l2_reg += torch.norm(param)
        return self.alpha * l2_reg
    
class Encoder_Ridge(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout
       

    def forward(self, x, data_key=None, detach_core=False, fake_relu = False, **kwargs):
        x = self.core(x).view(x.shape[0], -1)
        if detach_core:
            x = x.detach()
        if "sample" in kwargs:
            x = self.readout(x,  sample=kwargs["sample"])
        else:
            x = self.readout(x)
        return x 
    

class Encoder_Caption_semantic(nn.Module):  
    def __init__(self, core, readout, output_dim, text_dim):
        super().__init__()
        self.core = core
        self.readout = readout
        self.linear = nn.Linear(output_dim+text_dim, output_dim)

    def forward(self, x, img, single_x, data_key=None, detach_core=False, fake_relu = False, **kwargs):
        feats = self.core(x)
        if detach_core:
            feats = feats.detach()
        
        x = self.readout(feats, img)
        x = torch.cat((x, single_x), dim=1)
        x = self.linear(x)
        return x
