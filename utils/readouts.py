import torch
from utils.model_utils import *
from utils.train_utils import *
from torch import nn
from torch.autograd import Variable 
from torchvision.models import resnet50
from torch.nn.parameter import Parameter

class SemanticSpatialTransformer(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(self, in_shape, outdims, bias, spatial_dim = 28, return_att = False, mode = 'affine', normalize=True, init_noise=1e-3, constrain_pos=False, **kwargs):
        super().__init__()
        self.mode = mode
        self.in_shape = in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.return_att = return_att
        self.spatial_dim = spatial_dim
        c, w, h = in_shape
        # w = spatial_dim
        # h = spatial_dim
        self.channels = c
        self.spatial = Parameter(torch.Tensor(self.outdims, w, h))
        self.features = Parameter(torch.Tensor(self.outdims, c))
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
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              (-torch.sum((xy_grid - mean)**2.0, dim=-1) /\
                              (2*variance)).float()
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.outdims, 1, 1, 1)

        self.gaussian_filter = torch.nn.Conv2d(in_channels=self.outdims, out_channels=self.outdims,
                                    kernel_size=kernel_size, groups=self.outdims, bias=False)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False
        self.attention_filter = torch.nn.Conv2d(self.channels, self.outdims, kernel_size=5, padding = 1, bias=True)
        self.attention_filter.weight.requires_grad = True 
        self.attention_filter.bias.requires_grad = True 
        
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
        self.fc_enc = nn.Sequential(
            nn.Linear(4 * self.final_dim * self.final_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2 * self.channels)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_enc[2].weight.data.zero_()
        self.fc_enc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).repeat(self.channels))

    @property
    def normalized_spatial(self):
        positive(self.spatial)
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
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
        self.spatial.data.normal_(0, self.init_noise)
        self.features.data.normal_(0, self.init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, img, shift=None):
        
        B, c, w, h = x.size()
        ## localisation network
        spatial_mask =  self.normalized_spatial[None].repeat(B, 1, 1, 1)  # repeat along batch dimension
        spatial_mask = spatial_mask.view(B*self.outdims, 1, w, h) 
        xs = self.localization(img)
        xs = xs.view(-1, 4 * self.final_dim * self.final_dim)

        theta = self.fc_loc(xs)
        theta = theta.view(B*self.outdims, 6)

        theta_enc = self.fc_enc(xs)
        theta_enc = theta_enc.view(B*self.channels, 6)

        if self.mode == 'affine':
            theta1 = theta.view(-1, 2, 3)
            theta2 = theta_enc.view(-1, 2, 3)
        else: 
            theta1 = Variable( torch.zeros([B*self.outdims, 2, 3], dtype=torch.float32, device=xs.get_device()), requires_grad=True)
            theta1 = theta1 + 0
            theta1[:,0,0] = 1.0
            theta1[:,1,1] = 1.0
            if self.mode == 'translation':
                theta1[:,0,2] = theta[:,0]
                theta1[:,1,2] = theta[:,1]
            elif self.mode == 'rotation':
                angle = theta[:,0]
                theta1[:,0,0] = torch.cos(angle)
                theta1[:,0,1] = -torch.sin(angle)
                theta1[:,1,0] = torch.sin(angle)
                theta1[:,1,1] = torch.cos(angle)
            elif self.mode == 'scale':
                theta1[:,0,0] = theta[:,0]
                theta1[:,1,1] = theta[:,1]
            elif self.mode == 'shear':
                theta1[:,0,1] = theta[:,0]
                theta1[:,1,0] = theta[:,1]
            elif self.mode == 'rotation_scale':
                angle = theta[:,0]
                theta1[:,0,0] = torch.cos(angle) * theta[:,1]
                theta1[:,0,1] = -torch.sin(angle)
                theta1[:,1,0] = torch.sin(angle)
                theta1[:,1,1] = torch.cos(angle) * theta[:,2]
            elif self.mode == 'translation_scale':
                theta1[:,0,2] = theta[:,0]
                theta1[:,1,2] = theta[:,1]
                theta1[:,0,0] = theta[:,2]
                theta1[:,1,1] = theta[:,3]
            elif self.mode == 'rotation_translation':
                angle = theta[:,0]
                theta1[:,0,0] = torch.cos(angle)
                theta1[:,0,1] = -torch.sin(angle)
                theta1[:,1,0] = torch.sin(angle)
                theta1[:,1,1] = torch.cos(angle)
                theta1[:,0,2] = theta[:,1]
                theta1[:,1,2] = theta[:,2]
            elif self.mode == 'rotation_translation_scale':
                angle = theta[:,0]
                theta1[:,0,0] = torch.cos(angle) * theta[:,3]
                theta1[:,0,1] = -torch.sin(angle)
                theta1[:,1,0] = torch.sin(angle)
                theta1[:,1,1] = torch.cos(angle) * theta[:,4]
                theta1[:,0,2] = theta[:,1]
                theta1[:,1,2] = theta[:,2]
                
        ## grid consists of the xy coordinates of the image. In case theta is an identity transform, it will range from -a to a. 
        ## However, they will change accordingly with theta. These are then applied to the actual image in grid_sample via 
        ## bilinear interpolation

        ## Applying transformations on the spatial weights
        grid = F.affine_grid(theta1, spatial_mask.size()) ## grid generator
        spatial_mask = F.grid_sample(spatial_mask, grid) ## sampler
        spatial_mask = spatial_mask.view(B, self.outdims, w, h)

        ## Applying transformations on the encoding inputs
        x = x.view(B*self.channels, 1, w, h) 
        grid = F.affine_grid(theta2, x.size()) ## grid generator
        x = F.grid_sample(x, grid) ## sampler
        x = x.view(B, self.channels, w, h)
        
        y = torch.einsum('ncwh,nowh->nco', x, spatial_mask) 
        y = torch.einsum('nco,oc->no', y, self.features)
        if self.bias is not None:
            y = y + self.bias
        if self.return_att:
            return y, spatial_mask
        else:
            return y

    def __repr__(self):
        return ('normalized ' if self.normalize else '') + \
               self.__class__.__name__ + \
               ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(
            self.outdims) + ')'    

class GlobalLinear(nn.Module):
    """
    Global average pooling layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(self, in_shape, outdims, bias, init_noise=1e-3, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        c, w, h = in_shape
        self.channels = c
  
        self.features = Parameter(torch.Tensor(self.outdims, c))
        self.init_noise = init_noise
 
        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.initialize()
 

    def initialize(self):

        self.features.data.normal_(0, self.init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, shift=None):
        
        #x = torch.nn.AdaptiveAvgPool2d((10,10))(x)
       
        spatial_mean = x.mean((2,3))[:,:,None]
        
        y = spatial_mean.repeat(1,1,self.outdims)
        y = torch.einsum('nco,oc->no', y, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        return ('normalized ' if self.normalize else '') + \
               self.__class__.__name__ + \
               ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(
            self.outdims) + ')'    
    
    
class AttentionLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(self, in_shape, outdims, bias, return_att = False, normalize=True, init_noise=1e-3, constrain_pos=False, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.return_att = return_att
        c, w, h = in_shape
        self.channels = c
        #self.spatial = Parameter(torch.Tensor(self.outdims, w, h))
        self.features = Parameter(torch.Tensor(self.outdims, c))
        self.init_noise = init_noise
        self.constrain_pos = constrain_pos
        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.initialize()
        
        # Set these to whatever you want for your gaussian filter
        kernel_size = 5 #15
        sigma = 1 #3

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              (-torch.sum((xy_grid - mean)**2.0, dim=-1) /\
                              (2*variance)).float()
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

        self.gaussian_filter = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=kernel_size, groups=1, bias=False)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False
        self.attention_filter = torch.nn.Conv2d(self.channels, 1, kernel_size=3, padding = 1, bias=True)
        self.attention_filter.weight.requires_grad = True 
        self.attention_filter.bias.requires_grad = True 

    @property
    def normalized_spatial(self):
        positive(self.spatial)
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
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
        #self.spatial.data.normal_(0, self.init_noise)
        self.features.data.normal_(0, self.init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, shift=None):
        att = self.attention_filter(x)
        att = self.gaussian_filter(att)
        
        output = torch.flatten(att, start_dim=1)
        output = torch.nn.Softmax()(output)
        output = torch.reshape(output, att.size())
        
        if self.constrain_pos:
            positive(self.features)
            positive(self.normalized_spatial)
        x = torch.nn.AdaptiveAvgPool2d(tuple(att.size()[2:]))(x)
        output = output.repeat(1, self.channels, 1, 1) ### Same attention map - could potential diversify
        spatial_mean = (output*x).sum((2,3))[:,:,None]
        
        y = spatial_mean.repeat(1,1,self.outdims)#torch.einsum('ncwh,owh->nco', x, output) #self.normalized_spatial) #ncwh *nwho
        y = torch.einsum('nco,oc->no', y, self.features)
        if self.bias is not None:
            y = y + self.bias
        if self.return_att:
            return y, output
        else:
            return y

    def __repr__(self):
        return ('normalized ' if self.normalize else '') + \
               self.__class__.__name__ + \
               ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(
            self.outdims) + ')'    