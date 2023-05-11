import torch as t
import torch.nn as nn
import torchvision as tv

# deformable convolution
from model.dcn.deform_conv_v2 import DeformableConv2d
from utils.config import opt
from model.utils.misc import normal_init



# @TODO: add modulated deformable convolution
# if apply deformable convolution:
#     if modulated:
#         replace mdeconv@c3 ~ c5 + mdpool
#     else:
#         replace dconv@c3 ~ c5 + dpool

'''
    Load pretrained VGG16 model and replace 
    all dense layers with conv layers '''
def load_vgg16_extractor(pretrained=True, load_basic=False):

    vgg16 = tv.models.vgg16(pretrained=pretrained)

    if load_basic:
        # drop last max pooling layer
        features = list(vgg16.features)[:-1]

        # keep C1, C2 unchanged
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        # convert C5 layer in VGG16 to deformable Convolution layer
        if opt.deformable:
            # get weight and bias from pretrained model
            conv5_1_weight = vgg16.features[24].weight
            conv5_1_bias = vgg16.features[24].bias
            conv5_2_weight = vgg16.features[26].weight
            conv5_2_bias = vgg16.features[26].bias
            conv5_3_weight = vgg16.features[28].weight
            conv5_3_bias = vgg16.features[28].bias

            
            vgg16.features[24] = DeformableConv2d(512,512, (3,3),1, 1)
            vgg16.features[26] = DeformableConv2d(512,512, (3,3),1, 1)
            vgg16.features[28] = DeformableConv2d(512,512, (3,3),1, 1)

            #load pretrained weight and bias to deformable convolution
            vgg16.features[24].weight = nn.Parameter(conv5_1_weight.view(512,512,3,3))
            vgg16.features[24].bias = nn.Parameter(conv5_1_bias)
            vgg16.features[26].weight = nn.Parameter(conv5_2_weight.view(512,512,3,3))
            vgg16.features[26].bias = nn.Parameter(conv5_2_bias)
            vgg16.features[28].weight = nn.Parameter(conv5_3_weight.view(512,512,3,3))
            vgg16.features[28].bias = nn.Parameter(conv5_3_bias)
            

            
        return nn.Sequential(*features)
    else:
        features = list(vgg16.features)
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        # convert C5 layer in VGG16 to deformable Convolution layer
        if opt.deformable:
            # get weight and bias from pretrained model
            conv5_1_weight = vgg16.features[24].weight
            conv5_1_bias = vgg16.features[24].bias
            conv5_2_weight = vgg16.features[26].weight
            conv5_2_bias = vgg16.features[26].bias
            conv5_3_weight = vgg16.features[28].weight
            conv5_3_bias = vgg16.features[28].bias

            
            vgg16.features[24] = DeformableConv2d(512,512, (3,3),1, 1)
            vgg16.features[26] = DeformableConv2d(512,512, (3,3),1, 1)
            vgg16.features[28] = DeformableConv2d(512,512, (3,3),1, 1)

            #load pretrained weight and bias to deformable convolution
            vgg16.features[24].weight = nn.Parameter(conv5_1_weight.view(512,512,3,3))
            vgg16.features[24].bias = nn.Parameter(conv5_1_bias)
            vgg16.features[26].weight = nn.Parameter(conv5_2_weight.view(512,512,3,3))
            vgg16.features[26].bias = nn.Parameter(conv5_2_bias)
            vgg16.features[28].weight = nn.Parameter(conv5_3_weight.view(512,512,3,3))
            vgg16.features[28].bias = nn.Parameter(conv5_3_bias)

        # add extra convolution layer as (C6)
        conv6_1 = nn.Conv2d(512, 1024, 3, 1, padding=3, dilation=3)
        conv6_2 = nn.Conv2d(1024, 1024, 1, 1)

        # reshape pretrained weight
        conv6_1_weight = vgg16.classifier[0].weight.view(4096, 512, 7, 7)
        conv6_1_bias = vgg16.classifier[0].bias

        conv6_2_weight = vgg16.classifier[3].weight.view(4096, 4096, 1, 1)
        conv6_2_bias = vgg16.classifier[3].bias

        # subsampling weight
        conv6_1.weight = nn.Parameter(decimate(conv6_1_weight, m=[4, None, 3, 3]))
        conv6_1.bias = nn.Parameter(decimate(conv6_1_bias, m=[4]))

        conv6_2.weight = nn.Parameter(decimate(conv6_2_weight, m=[4, 4, None, None]))
        conv6_2.bias = nn.Parameter(decimate(conv6_2_bias, m=[4]))

        features += [conv6_1, nn.ReLU(True), conv6_2, nn.ReLU(True)]

        return nn.Sequential(*features)

def load_vgg16_classifier(pretrained=True,load_basic=False):
    if load_basic:
        vgg16 = tv.models.vgg16(pretrained=pretrained)
        top_layer = list(vgg16.classifier)[:6]
        del top_layer[5]
        del top_layer[2]
        return nn.Sequential(*top_layer)
    else:
        return nn.Sequential(nn.Linear(7 * 7 * 256, 1024),
                                    nn.ReLU(True),
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(True))


def decimate(tensor, m):
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(
                dim=d, index=t.arange(start=0, end=tensor.size(d), step=m[d]).long()
            )

    return tensor
