import torch
from utils.model_utils import *
from utils.train_utils import *
from torch import nn
from torch.autograd import Variable 
from torchvision.models import resnet50
from torch.nn.parameter import Parameter

class SemanticSpatialTransformerTextImage(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(self, in_shape_img, in_shape_text, outdims, bias, spatial_dim = 28, return_att = False, mode = 'affine', normalize=True, init_noise=1e-3, constrain_pos=False, **kwargs):
        super().__init__()
        self.mode = mode
        self.in_shape_img = in_shape_img
        self.in_shape_text = in_shape_text
        self.outdims = outdims
        self.normalize = normalize
        self.return_att = return_att
        self.spatial_dim = spatial_dim
        c_img, w_img, h_img = in_shape_img
        c_txt, w_txt, h_txt = in_shape_text
        # w = spatial_dim
        # h = spatial_dim
        self.channels_img = c_img
        self.channels_txt = c_txt
        self.spatial_img = Parameter(torch.Tensor(self.outdims, w_img, h_img))
        self.features = Parameter(torch.Tensor(self.outdims, c_img + c_txt))
        self.spatial_txt = Parameter(torch.Tensor(self.outdims, w_txt, h_txt))
        self.init_noise = init_noise
        self.constrain_pos = constrain_pos
        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.initialize()
        
        # Set these to whatever you want for your gaussian filter
        kernel_size = 15
        sigma = 3

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()

        # Spatial transformer localization-network
        semantic = resnet50(pretrained = True)
        for param in semantic.parameters():
                   param.requires_grad = False
        self.localization = nn.Sequential(
            *list(semantic.children())[:-2],
            nn.Conv2d(2048, 4, kernel_size=1),
          #  nn.MaxPool2d(2, stride=2),
           # nn.ReLU(True)
        )
        #self.localization = nn.Sequential(*list(semantic.children())[:-2])
        
        self.final_dim = 7
        
        # Regressor for the 3 * 2 affine matrix (spatial weights)
        self.fc_loc = nn.Sequential(
            nn.Linear(4 * self.final_dim * self.final_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2 * self.outdims)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).repeat(self.outdims))

        # Regressor for the 3 * 2 affine matrix (encoder output)
        self.fc_enc_img = nn.Sequential(
            nn.Linear(4 * self.final_dim * self.final_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2 * self.channels_img)
        )
        self.fc_enc_txt = nn.Sequential(
            nn.Linear(4 * self.final_dim * self.final_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2 * self.channels_txt)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_enc_img[2].weight.data.zero_()
        self.fc_enc_img[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).repeat(self.channels_img))

        self.fc_enc_txt[2].weight.data.zero_()
        self.fc_enc_txt[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).repeat(self.channels_txt))

    @property
    def normalized_spatial_img(self):
        positive(self.spatial_img)
        if self.normalize:
            norm = self.spatial_img.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial_img) + 1e-6
            weight = self.spatial_img / norm
        else:
            weight = self.spatial_img
        return weight
    
    @property
    def normalized_spatial_txt(self):
        positive(self.spatial_txt)
        if self.normalize:
            norm = self.spatial_txt.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial_txt) + 1e-6
            weight = self.spatial_txt / norm
        else:
            weight = self.spatial_txt
        return weight

    # TODO: Fix weight property -> self.positive is not defined
    @property
    def weight(self):
        if self.positive:
            positive(self.features)
        n = self.outdims
        c, w, h = self.in_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

    def l1(self, average=False):
        n = self.outdims
        c, w, h = self.in_shape
        ret = (self.normalized_spatial.view(self.outdims, -1).abs().sum(dim=1, keepdim=True)
               * self.features.view(self.outdims, -1).abs().sum(dim=1)).sum()
        if average:
            ret = ret / (n * c * w * h)
        return ret

    def initialize(self):
        self.spatial_img.data.normal_(0, self.init_noise)
        self.spatial_txt.data.normal_(0, self.init_noise)
        self.features.data.normal_(0, self.init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x_img, x_txt, img, txt, shift=None):
        
        B, c_img, w_img, h_img = x_img.size()
        B, c_txt, w_txt, h_txt = x_txt.size()

        ## localisation network for image weights
        spatial_mask_img =  self.normalized_spatial_img[None].repeat(B, 1, 1, 1)  # repeat along batch dimension
        spatial_mask_img = spatial_mask_img.view(B*self.outdims, 1, w_img, h_img) 
        xs_img = self.localization(img)
        xs_img = xs_img.view(-1, 4 * self.final_dim * self.final_dim)

        theta = self.fc_loc(xs_img)
        theta = theta.view(B*self.outdims, 6)

        theta_enc = self.fc_enc_img(xs_img)
        theta_enc = theta_enc.view(B*self.channels_img, 6)

        theta1 = theta.view(-1, 2, 3)
        theta2 = theta_enc.view(-1, 2, 3)

        grid = F.affine_grid(theta1, spatial_mask_img.size()) ## grid generator
        spatial_mask_img = F.grid_sample(spatial_mask_img, grid) ## sampler
        spatial_mask_img = spatial_mask_img.view(B, self.outdims, w_img, h_img)

        x_img = x_img.view(B*self.channels_img, 1, w_img, h_img) 
        grid = F.affine_grid(theta2, x_img.size()) ## grid generator
        x_img = F.grid_sample(x_img, grid) ## sampler
        x_img = x_img.view(B, self.channels_img, w_img, h_img)

        y1 = torch.einsum('ncwh,nowh->nco', x_img, spatial_mask_img) 

        ## localisation network for text caption weights
        spatial_mask_txt =  self.normalized_spatial_txt[None].repeat(B, 1, 1, 1)  # repeat along batch dimension
        spatial_mask_txt = spatial_mask_txt.view(B*self.outdims, 1, w_txt, h_txt) 
        xs_txt = self.localization(img)
        xs_txt = xs_txt.view(-1, 4 * self.final_dim * self.final_dim)

        theta = self.fc_loc(xs_txt)
        theta = theta.view(B*self.outdims, 6)

        theta_enc = self.fc_enc_txt(xs_txt)
        theta_enc = theta_enc.view(B*self.channels_txt, 6)

        theta1 = theta.view(-1, 2, 3)
        theta2 = theta_enc.view(-1, 2, 3)

        grid = F.affine_grid(theta1, spatial_mask_txt.size()) ## grid generator
        spatial_mask_txt = F.grid_sample(spatial_mask_txt, grid) ## sampler
        spatial_mask_txt = spatial_mask_txt.view(B, self.outdims, w_txt, h_txt)

        x_txt = x_txt.view(B*self.channels_txt, 1, w_txt, h_txt) 
        grid = F.affine_grid(theta2, x_txt.size()) ## grid generator
        x_txt = F.grid_sample(x_txt, grid) ## sampler
        x_txt = x_txt.view(B, self.channels_txt, w_txt, h_txt)

        y2 = torch.einsum('ncwh,nowh->nco', x_txt, spatial_mask_txt) 
        y = torch.cat((y1,y2),dim=1)
        y = torch.einsum('nco,oc->no', y, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        return ('normalized ' if self.normalize else '') + \
               self.__class__.__name__ + \
               ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(
            self.outdims) + ')'  