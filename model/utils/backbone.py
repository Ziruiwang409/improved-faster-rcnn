import torch as t
import torch.nn as nn
import torchvision as tv

# deformable convolution
from model.dcnv2.dcn_v2 import dcn_v2_conv, DCNv2, DCN
from model.dcnv2.dcn_v2 import dcn_v2_pooling, DCNv2Pooling, DCNPooling



# @TODO: add modulated deformable convolution
# if apply deformable convolution:
#     if modulated:
#         replace mdeconv@c3 ~ c5 + mdpool
#     else:
#         replace dconv@c3 ~ c5 + dpool

'''
    Load pretrained VGG16 model and replace 
    all dense layers with conv layers '''
def load_vgg16_extractor(pretrained=True, deformable=False, load_basic=False):

    vgg16 = tv.models.vgg16(pretrained=pretrained)

    # if deformable:
    if deformable:
        '''
            replace dconv@c3 ~ c5
        '''
        for i, layer in enumerate(vgg16.features):
            if i >= 10 and isinstance(layer, nn.Conv2d):
                vgg16.features[i] = DCN(in_channels=layer.in_channels, 
                                          out_channels=layer.out_channels, 
                                          kernel_size=layer.kernel_size, 
                                          stride=layer.stride, 
                                          padding=layer.padding, 
                                          deformable_groups=2).cuda()

    if load_basic:
        # drop last max pooling layer
        features = list(vgg16.features)[:-1]

        # keep first 10 layers fixed
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False
            
        return nn.Sequential(*features)
    else:
        features = list(vgg16.features)
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        conv6 = nn.Conv2d(512, 1024, 3, 1, padding=3, dilation=3)
        conv7 = nn.Conv2d(1024, 1024, 1, 1)

        # reshape pretrained weight
        conv6_weight = vgg16.classifier[0].weight.view(4096, 512, 7, 7)
        conv6_bias = vgg16.classifier[0].bias

        conv7_weight = vgg16.classifier[3].weight.view(4096, 4096, 1, 1)
        conv7_bias = vgg16.classifier[3].bias

        # subsampling weight
        conv6.weight = nn.Parameter(decimate(conv6_weight, m=[4, None, 3, 3]))
        conv6.bias = nn.Parameter(decimate(conv6_bias, m=[4]))

        conv7.weight = nn.Parameter(decimate(conv7_weight, m=[4, 4, None, None]))
        conv7.bias = nn.Parameter(decimate(conv7_bias, m=[4]))

        features += [conv6, nn.ReLU(True), conv7, nn.ReLU(True)]

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
