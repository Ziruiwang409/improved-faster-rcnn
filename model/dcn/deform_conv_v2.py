import torch
import torchvision.ops
from torch import nn

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        # calculate offset for deformable convolution
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        # caluclate masks for deformable convolution (NOTE: mask is set to None for DCNv1)
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=None,
                                          stride=self.stride,
                                          )
        return x

class DeformableRoIPool(nn.Module):
    def __init__(self, 
                 output_size,
                 spatial_scale):
        
        super(DeformableRoIPool, self).__init__()

        assert type(output_size) == tuple or type(output_size) == int
        output_size = output_size if type(output_size) == int else output_size[0]

        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.offset_fc = nn.Sequential(nn.Linear(self.output_size * self.output_size * 512, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, self.output_size * self.output_size * 512)
                                       )
        # initialize weight for fc layer
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()
    
    def forward(self, input, rois):
        # 1. RoI pooling generates the pooled feature map
        pooled_features = torchvision.ops.roi_pool(input, 
                                                   rois, 
                                                   output_size=self.output_size, 
                                                   spatial_scale=self.spatial_scale,
                                                   )
        # 2. a fc layer generates the normalized offsets
        offset_feature_map = self.offset_fc(pooled_features.view(pooled_features.shape[0],-1))

        # 3. transform the offset by element-wise product with the RoI's width and height
        # reshape
        offset_feature_map = offset_feature_map.view(-1, 512, self.output_size, self.output_size)
        offset_feature_map[:, 0, :, :] = offset_feature_map[:, 0, :, :] * rois[:, 3].unsqueeze(1).unsqueeze(2)
        offset_feature_map[:, 1, :, :] = offset_feature_map[:, 1, :, :] * rois[:, 4].unsqueeze(1).unsqueeze(2)

        return offset_feature_map

