import torch

#################################################################
# initialize weights
def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    m.weight.data.normal_(mean, stddev) 
    m.bias.data.zero_()

#################################################################
# bounding box regression loss

def bbox_regression_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

#################################################################
# FPN helper functions

# NOTE: Helper function for deciding the feature level of each RoI
def assign_feature_level(roi, k0, low, high):
    h = roi.data[:, 2] - roi.data[:, 0] + 1
    w = roi.data[:, 3] - roi.data[:, 1] + 1
    # get feature level based on formula: 
    k = torch.log2(torch.sqrt(h * w) / 224.) + k0

    # get lower & upper limit of feature levels
    k = torch.round(k)
    k[k < low] = low
    k[k > high] = high

    return k