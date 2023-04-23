import torch as t
import torch.nn as nn
import torchvision as tv
from model.deform_conv_v2 import DeformConv2d


# @TODO: add modulated deformable convolution
# if apply deformable convolution:
#     if modulated:
#         replace mdeconv@c3 ~ c5 + mdpool
#     else:
#         replace dconv@c3 ~ c5 + dpool

'''
    Load pretrained VGG16 model and replace 
    all dense layers with conv layers '''
def load_vgg16(pretrained=True, deformable=False, modulated=False):

    model = tv.models.vgg16(pretrained=pretrained)

    features = list(model.features)
    # train on conv4, conv5 in 
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False


    conv6 = nn.Conv2d(512, 1024, 3, 1, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, 1, 1)

    # reshape pretrained weight
    conv6_weight = model.classifier[0].weight.view(4096, 512, 7, 7)
    conv6_bias = model.classifier[0].bias

    conv7_weight = model.classifier[3].weight.view(4096, 4096, 1, 1)
    conv7_bias = model.classifier[3].bias

    # subsampling weight
    conv6.weight = nn.Parameter(decimate(conv6_weight, m=[4, None, 3, 3]))
    conv6.bias = nn.Parameter(decimate(conv6_bias, m=[4]))

    conv7.weight = nn.Parameter(decimate(conv7_weight, m=[4, 4, None, None]))
    conv7.bias = nn.Parameter(decimate(conv7_bias, m=[4]))

    features += [conv6, nn.ReLU(True), conv7, nn.ReLU(True)]

    if deformable and not modulated:
        '''
            replace dconv@c3 ~ c5
        '''
        features[10] = DeformConv2d(128,256,3,1,1)
        features[12] = DeformConv2d(256,256,3,1,1)
        features[14] = DeformConv2d(256,256,3,1,1)
        features[17] = DeformConv2d(256,512,3,1,1)
        features[19] = DeformConv2d(512,512,3,1,1)
        features[21] = DeformConv2d(521,512,3,1,1)
        features[24] = DeformConv2d(521,512,3,1,1)
        features[26] = DeformConv2d(521,512,3,1,1)
        features[28] = DeformConv2d(521,512,3,1,1)
        # TODO: replace Conv2d with DCN2d for upsampling layer?
        features[31] = DeformConv2d(521,1024,3,1,1)
        features[33] = DeformConv2d(1024,1024,3,1,1)
    elif deformable and modulated:
        '''
            replace mdconv@c3 ~ c5
        '''
        features[10] = DeformConv2d(128,256,3,1,1, modulation=True)
        features[12] = DeformConv2d(256,256,3,1,1, modulation=True)
        features[14] = DeformConv2d(256,256,3,1,1, modulation=True)
        features[17] = DeformConv2d(256,512,3,1,1, modulation=True)
        features[19] = DeformConv2d(512,512,3,1,1, modulation=True)
        features[21] = DeformConv2d(521,512,3,1,1, modulation=True)
        features[24] = DeformConv2d(521,512,3,1,1, modulation=True)
        features[26] = DeformConv2d(521,512,3,1,1, modulation=True)
        features[28] = DeformConv2d(521,512,3,1,1, modulation=True)
        # TODO: replace Conv2d with DCN2d for upsampling layer?
        features[31] = DeformConv2d(521,1024,3,1,1, modulation=True)
        features[33] = DeformConv2d(1024,1024,3,1,1, modulation=True)

    return nn.Sequential(*features)


def decimate(tensor, m):
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(
                dim=d, index=t.arange(start=0, end=tensor.size(d), step=m[d]).long()
            )

    return tensor
"""
######################################
No deformable
######################################

Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace=True)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace=True)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace=True)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace=True)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace=True)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace=True)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace=True)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (31): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=(3, 3))
  (32): ReLU(inplace=True)
  (33): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
  (34): ReLU(inplace=True)
)
"""

"""
######################################
Deformable = True, Modulated = False 
######################################

Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(128, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (11): ReLU(inplace=True)
  (12): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (13): ReLU(inplace=True)
  (14): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (15): ReLU(inplace=True)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (18): ReLU(inplace=True)
  (19): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(512, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (20): ReLU(inplace=True)
  (21): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(521, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(521, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (22): ReLU(inplace=True)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(521, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(521, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (25): ReLU(inplace=True)
  (26): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(521, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(521, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (27): ReLU(inplace=True)
  (28): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(521, 512, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(521, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (29): ReLU(inplace=True)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (31): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(521, 1024, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(521, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (32): ReLU(inplace=True)
  (33): DeformConv2d(
    (zero_padding): ZeroPad2d((1, 1, 1, 1))
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(3, 3), bias=False)
    (p_conv): Conv2d(1024, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (34): ReLU(inplace=True)
)
"""

