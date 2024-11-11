import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import torch
from utils.train_utils import *
from utils.model_utils import *
from utils.dataloaders import Dataset_visual
from models.model_v1 import RCNNV1
from models.model_v2 import RCNNV2
from models.model_v4 import RCNNV4
from models.model_it import RCNNIT
from neuralpredictors.layers.readouts import SpatialXFeatureLinear
from utils.readouts import SemanticSpatialTransformer
from models.models_brain import C8SteerableCNN
from torch.autograd import Variable
from itertools import repeat
from utils.utils import save_checkpoint
from tqdm import tqdm
import argparse
import logging

v1v_neurons = 2058
v1d_neurons = 2509
v1l_neurons = 4567
v2v_neurons = 2406
v2d_neurons = 1957
v2l_neurons = 4363
v4_neurons = 2039

def get_neurons(brain_region, layer):
    neurons = None
    if layer == 1:
        if brain_region == 'ventral_visual_data':
            neurons = v1v_neurons
        elif brain_region == 'dorsal_visual_data':
            neurons = v1d_neurons
        elif brain_region == 'lateral_visual_data':
            neurons = v1l_neurons
    elif layer == 2:
        if brain_region == 'ventral_visual_data':
            neurons = v2v_neurons
        elif brain_region == 'dorsal_visual_data':
            neurons = v2d_neurons
        elif brain_region == 'lateral_visual_data':
            neurons = v2l_neurons
    elif layer == 4:
        neurons = v4_neurons
    return neurons

def load_submodels(brain_region, layer, saved_model, n_feats, core_v1v=None, core_v2v=None):
    neurons = get_neurons(brain_region, layer)
    core, readout = None, None
    if layer == 1:
        core = RCNNV1(n_feats = n_feats)
        readout = SpatialXFeatureLinear(core(torch.randn(1, 3, 224, 224)).size()[1:], neurons,  bias = True)  
    elif layer == 2:
        core = RCNNV2(n_feats = n_feats)
        readout = SpatialXFeatureLinear(core(core_v1v(torch.randn(1, 3, 224, 224))).size()[1:], neurons,  bias = True) 
    elif layer == 4:
        core = RCNNV4(n_feats = n_feats)
        readout = SpatialXFeatureLinear(core(core_v2v(core_v1v(torch.randn(1, 3, 224, 224)))).size()[1:], neurons,  bias = True) 
    model = Encoder(core, readout)
    model.train()
    checkpoint = torch.load(saved_model)
    checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    print("model loaded :: ", layer)
    return model, core, readout

def load_layered_IT_model(n_neurons, saved_model, core_v1, core_v2, core_v4, readout = 'semantic_transformer'):
    n_feats = 48
    core = RCNNIT(n_feats = n_feats)
    if readout == 'semantic_transformer':
        readout = SemanticSpatialTransformer(core(core_v4(core_v2(core_v1(torch.randn(1, 3, 224, 224))))).size()[1:], n_neurons,  bias = True)  
        model = Encoder_semantic(core, readout)
    elif readout == 'spatial_linear':
        readout = SpatialXFeatureLinear(core(core_v4(core_v2(core_v1(torch.randn(1, 3, 224, 224))))).size()[1:], n_neurons,  bias = True)  
        model = Encoder(core, readout)
    model.train()
    print("saved model ::", saved_model)
    checkpoint = torch.load(saved_model)
    checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    print("model loaded :: IT")
    return model, core, readout

def load_vanilla_IT_model(n_neurons, saved_model, readout = 'semantic_transformer'):
    n_feats = 48
    core = C8SteerableCNN(n_feats = n_feats)
    if readout == 'spatial_linear':
        print("spatial linear readout")
        readout = SpatialXFeatureLinear(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)  
        model = Encoder(core, readout)
    elif readout == 'semantic_transformer':
        print("semantic transformer readout")
        readout = SemanticSpatialTransformer(core(torch.randn(1, 3, 224, 224)).size()[1:], n_neurons,  bias = True)  
        model = Encoder_semantic(core, readout)
    model.train()
    print("saved model ::", saved_model)
    checkpoint = torch.load(saved_model)
    checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    print("model loaded :: VanillaIT")
    return model, core, readout