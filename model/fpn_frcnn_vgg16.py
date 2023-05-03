# PyTorch Pakages
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision as tv

# Faster R-CNN Packages
from model.frcnn_bottleneck import FasterRCNNBottleneck
from model.utils.backbone import load_vgg16_extractor,load_vgg16_classifier
from model.rpn.region_proposal_network import FPNBasedRPN
from model.utils.misc import normal_init, assign_feature_level

# Deformable Convolution
from model.dcnv2.dcn_v2 import dcn_v2_conv, DCNv2, DCN
from model.dcnv2.dcn_v2 import dcn_v2_pooling, DCNv2Pooling, DCNPooling


# Other Utils
from utils.config import opt
from utils import array_tool as at



class FPNFasterRCNNVGG16(FasterRCNNBottleneck):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    def __init__(self,n_fg_class=20):
        # feature extraction (Backbone CNN: VGG16)
        extractor = load_vgg16_extractor(pretrained=True,deformable=opt.deformable, load_basic=False)
        super(FPNFasterRCNNVGG16, self).__init__(
            n_classes=n_fg_class + 1,   # +1 for background
            extractor = extractor,     # feature extraction
            rpn=FPNBasedRPN(scales=[64, 128, 256, 512],
                            ratios=[0.5, 1, 2],
                            rpn_conv=nn.Conv2d(256, 512, 3, 1, 1),
                            rpn_loc=nn.Conv2d(512, 3 * 4, 1, 1),
                            rpn_score=nn.Conv2d(512, 3 * 2, 1, 1)),
            predictor=load_vgg16_classifier(load_basic=False),  # feature pooling and prediction 
            loc=nn.Linear(1024, (n_fg_class + 1) * 4),
            score=nn.Linear(1024, n_fg_class + 1),
            spatial_scale=[1/4.,1/8.,1/16.,1/32.],
            pooling_size=7,
            roi_sigma=opt.roi_sigma)
        normal_init(self.predictor[0], 0, 0.01)
        normal_init(self.predictor[2], 0, 0.01)
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # FPN parameters
        self.k0 = 5

        # initialize the parameters of RPN
        # 1. feature extraction layers
        self.extraction_layer = [15, 22, 29, 34]
        # 2. lateral connection layers
        self.lateral_layer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.lateral_layer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral_layer3 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral_layer4 = nn.Conv2d(256, 256, 1, 1, 0)

        normal_init(self.lateral_layer1, 0, 0.01)
        normal_init(self.lateral_layer2, 0, 0.01)
        normal_init(self.lateral_layer3, 0, 0.01)
        normal_init(self.lateral_layer4, 0, 0.01)

        # smooth layer
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        normal_init(self.smooth1, 0, 0.01)
        normal_init(self.smooth2, 0, 0.01)
        normal_init(self.smooth3, 0, 0.01)

    # bilinear interpolation upsampling
    def bilinear_interpolate(self, x, y):
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y
    
    # NOTE: inherent from FasterRCNN class
    def feature_extraction_layer(self, x):
        features = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            if i in self.extraction_layer:
                features.append(x)

        # top-down pathway & smoothing
        p6 = self.lateral_layer1(features[3])
        p5 = self.bilinear_interpolate(p6, self.lateral_layer2(features[2]))    # upsample
        p4 = self.bilinear_interpolate(p5, self.lateral_layer3(features[1]))    # upsample
        p3 = self.bilinear_interpolate(p4, self.lateral_layer4(features[0]))    # upsample
        
        p5 = self.smooth1(p5)
        p4 = self.smooth1(p4)
        p3 = self.smooth3(p3)

        return [p3, p4, p5, p6]
    
    # NOTE: inherent from FasterRCNN Class
    def roi_pooling_layer(self, feature, roi):
        roi = at.totensor(roi).float()
        # compute the lowest & highest level
        low = self.k0 - 2
        high = self.k0 + 1
        # compute feature level
        k = assign_feature_level(roi, self.k0, low, high)
        # possible starting level
        starting_lv = t.arange(low, high + 1)

        pooled_feats = []
        box_to_levels = []
        for i, l in enumerate(starting_lv):
            if (k == l).sum() == 0:
                continue

            level_idx = t.where(k == l)[0]
            box_to_levels.append(level_idx)
            
            # add batch index to roi
            rois = t.cat([t.zeros(level_idx.size(0), 1).cuda(), roi[level_idx]],dim=1)
            # change roi order to (batch_index, x1, y1, x2, y2)
            rois = rois[:, [0, 2, 1, 4, 3]].contiguous()

            if opt.deformable:
                dpooling = DCNPooling(spatial_scale=self.spatial_scale[i],
                                      pooled_size=self.pooling_size,
                                      output_dim=feature.shape[i][1],
                                      no_trans=False,
                                      group_size=1,
                                      trans_std=0.1,
                                      deform_fc_dim=1024).cuda()
                
                pooled_feat = dpooling(feature[i], rois)
            else:
                pooled_feat = tv.ops.roi_pool(feature[i],
                                              rois,
                                              self.pooling_size,
                                              self.spatial_scale[i])


            pooled_feats.append(pooled_feat)

        pooled_feats = t.cat(pooled_feats, dim=0)
        box_to_level = t.cat(box_to_levels, dim=0)
        idx_sorted, order = t.sort(box_to_level)
        pooled_feats = pooled_feats[order]

        return pooled_feats

    def bbox_regression_and_classification_layer(self, pooled_feature):
        
        # flatten roi pooled feature
        pooled_feature = pooled_feature.view(pooled_feature.shape[0], -1)

        # RCNN classifier
        fc9 = self.predictor(pooled_feature)

        # bbox regression & classification
        roi_loc = self.loc(fc9)
        roi_score = self.score(fc9)

        return roi_loc, roi_score

