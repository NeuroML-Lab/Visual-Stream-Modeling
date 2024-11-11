import shutil
import torch
import os
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import yaml
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])




class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints
    
    

def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor

def save_checkpoint(state, is_best, checkpoint, model_str):
       """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
       checkpoint + 'best.pth.tar'
       Args:
           state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
           is_best: (bool) True if it is the best model seen till now
           checkpoint: (string) folder where parameters are to be saved
       """
       existing_files = os.listdir(checkpoint)
       for f in existing_files:
        if f.startswith('last'):
            os.remove(os.path.join(checkpoint, f))
       filepath = os.path.join(checkpoint, 'last_%s.pth.tar'% model_str)
       if not os.path.exists(checkpoint):
        #    print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
           os.mkdir(checkpoint)
    #    else:
    #        print("Checkpoint Directory exists! ")
       torch.save(state, filepath)
       if is_best:
           shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        
def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        print("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'],strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return model #checkpoint    

class PoissonLoss(torch.nn.Module):
    def __init__(self, bias=1e-12, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        
        loss = (output - target * torch.log(output + self.bias))
        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)
        
class PoissonLoss3d(torch.nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        #_assert_no_grad(target)
        lag = target.size(1) - output.size(1)
        loss =  (output - target[:, lag:, :] * torch.log(output + self.bias))
        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)        