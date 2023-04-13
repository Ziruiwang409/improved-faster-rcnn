# PyTorch Pakages
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision as tv

# Faster R-CNN Packages
from model.faster_rcnn import FasterRCNN
from model.utils.backbone import load_vgg16
from model.region_proposal_network import FPNBasedRPN

# Other Utils
from utils.config import opt
from utils import array_tool as at


class FasterRCNNVGG16(FasterRCNN):
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
        extractor = load_vgg16(pretrained=True)
        super(FasterRCNNVGG16, self).__init__(
            n_classes=n_fg_class + 1,   # +1 for background
            extractor = extractor,     # feature extraction
            rpn=FPNBasedRPN(scales=[64, 128, 256, 512],
                            ratios=[0.5, 1, 2],
                            rpn_conv=nn.Conv2d(256, 512, 3, 1, 1),
                            rpn_loc=nn.Conv2d(512, 3 * 4, 1, 1),
                            rpn_score=nn.Conv2d(512, 3 * 2, 1, 1)),
            predictor=nn.Sequential(nn.Linear(7 * 7 * 256 * opt.n_features, 1024),
                                    nn.ReLU(True),
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(True)),  # feature pooling and prediction 
            loc=nn.Linear(1024, (n_fg_class + 1) * 4),
            score=nn.Linear(1024, n_fg_class + 1),
            spatial_scale=[1/4., 1/8., 1/16., 1/32.],
            pooling_size=7,
            roi_sigma=opt.roi_sigma)
        normal_init(self.predictor[0], 0, 0.01)
        normal_init(self.predictor[2], 0, 0.01)
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # FPN parameters
        self.k0 = 4

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
    def feature_extraction_module(self, x):
        features = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            if i in self.extraction_layer:
                features.append(x)

        # top-down pathway & smoothing
        p5 = self.lateral_layer1(features[3])
        p4 = self.bilinear_interpolate(p5, self.lateral_layer2(features[2]))    # upsample
        p4 = self.smooth1(p4)
        p3 = self.bilinear_interpolate(p4, self.lateral_layer3(features[1]))    # upsample
        p3 = self.smooth2(p3)
        p2 = self.bilinear_interpolate(p3, self.lateral_layer4(features[0]))    # upsample
        p2 = self.smooth3(p2)

        return [p2, p3, p4, p5]
    
    # NOTE: Helper function for deciding the feature level of each RoI
    def assign_feature_level(roi, n_features, k0, low, high):
        h = roi.data[:, 2] - roi.data[:, 0] + 1
        w = roi.data[:, 3] - roi.data[:, 1] + 1
        # get feature level based on formula: 
        k = t.log2(t.sqrt(h * w) / 224.) + k0

        # get lower & upper limit of feature levels
        if n_features == 1:
            level = t.round(level)
            level[level < low] = low
            level[level > high] = high
        elif n_features == 2:
            l1, l2, l3 = low, low + 1, low + 2
            level[level < l2] = l1
            level[(level >= l2) & (level < l3)] = l2
            level[level >= l3] = l3
        elif n_features == 3:
            limit = (low + high) / 2.
            level[level < limit] = low
            level[level >= limit] = low + 1
        else:
            raise NotImplementedError('Not implemented yet.')

        return level
    # NOTE: inherent from FasterRCNN Class
    def roi_pooling_module(self, features, roi):
        roi = at.totensor(roi).float()
        # n_features -> the number of features to use for RoI-Pooling
        #               not that of all features

        n_features = self.n_features
        # compute the lowest & highest level
        low = self.k0 - 2
        high = self.k0 + 1
        # compute feature level
        k = self.assign_feature_level(roi, n_features, self.k0, low, high)
        # possible starting level
        starting_lv = t.arange(low, high + 1)
        if n_features == 2:
            starting_lv = starting_lv[:-1]
        elif n_features == 3:
            starting_lv = starting_lv[:-2]
        # perform RoI-Pooling
        pooled_feats = []
        box_to_levels = []
        for i, l in enumerate(starting_lv):
            if (k == l).sum() == 0:
                continue

            level_idx = t.where(k == l)[0]
            box_to_levels.append(level_idx)

            index_and_roi = t.cat(
                [t.zeros(level_idx.size(0), 1).cuda(), roi[level_idx]],
                dim=1
            )
            # yx -> xy
            index_and_roi = index_and_roi[:, [0, 2, 1, 4, 3]].contiguous()

            pooled_feats_l = []
            for j in range(i, i + n_features):
                feat = tv.ops.roi_pool(
                    features[j],
                    index_and_roi,
                    self.pooling_size,
                    self.spatial_scale[j]
                )
                # feat -> n_roi_lx256x7x7
                pooled_feats_l.append(feat)

            pooled_feats.append(t.cat(pooled_feats_l, dim=1))

        pooled_feats = t.cat(pooled_feats, dim=0)
        box_to_level = t.cat(box_to_levels, dim=0)
        idx_sorted, order = t.sort(box_to_level)
        pooled_feats = pooled_feats[order]

        return pooled_feats

    def bbox_regression_and_classification_module(self, pooled_feature):
        
        # flatten roi pooled feature
        pooled_feature = pooled_feature.view(pooled_feature.shape[0], -1)

        # RCNN predictor
        fc9 = self.predictor(pooled_feature)

        # bbox regression & classification
        roi_loc = self.loc(fc9)
        roi_score = self.score(fc9)

        return roi_loc, roi_score


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
