import torch as t
from torch import nn
from torch.nn import functional as F

# Model Utils
from model.utils.bbox_tools import generate_anchors_fpn, generate_anchors
from model.rpn.proposal_tools import GenerateProposals, SampleTargetProposal, SampleTargetAnchor
from model.utils.misc import normal_init, bbox_regression_loss

# Other Utils
from utils.config import opt
import utils.array_tool as at


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features map extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features map from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """
    def __init__(self, scales, ratios):
        super(RegionProposalNetwork, self).__init__()
        self.scales = scales
        self.ratios = ratios
        self.generate_proposals_module = GenerateProposals(self)
        self.sample_proposal_module = SampleTargetProposal()
        self.sample_anchor_module = SampleTargetAnchor()
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.rpn_sigma = opt.rpn_sigma

    # NOTE: Implment in subclass (FPNBasedRPN)
    def forward(self, feature_maps, img_size, scale, gt_bbox, gt_label):
        raise NotImplementedError

class RPN(RegionProposalNetwork):
    def __init__(self, scales, ratios, rpn_conv, rpn_loc, rpn_score):
        super(RPN, self).__init__(scales, ratios)
        self.n_anchor = len(ratios) * len(scales)
        self.rpn_conv = rpn_conv
        self.rpn_loc = rpn_loc
        self.rpn_score = rpn_score
        normal_init(self.rpn_conv, 0, 0.01)
        normal_init(self.rpn_loc, 0, 0.01)
        normal_init(self.rpn_score, 0, 0.01)

    def forward(self, feature_maps, img_size, scale, gt_bbox, gt_label):
        n = 1  # batch size is always one

        h = F.relu(self.rpn_conv(feature_maps))

        loc = self.rpn_loc(h)   # (x,y,h,w)
        score = self.rpn_score(h)

        h, w = loc.shape[2:]

        # get bbox location and score
        loc = loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        score = score.permute(0, 2, 3, 1).contiguous()

        # softmax score
        softmax_score = F.softmax(score.view(n, h, w, self.n_anchor, 2), dim=4)
        fg_score = softmax_score[:, :, :, :, 1].contiguous().view(n, -1)

        shape = (h, w)
        score = score.view(n, -1, 2)

        # get feature stride
        stride = img_size[0] / h
        # NOTE: generate anchors (all layers)
        anchors = generate_anchors(self.scales, self.ratios, shape, stride)

        loc = loc[0]
        score = score[0]
        fg_score = fg_score[0]

        # get proposals given anchors and bbox offsets
        roi = self.generate_proposals_module(loc.cpu().data.numpy(),
                                             fg_score.cpu().data.numpy(),
                                             anchors,
                                             img_size,
                                             scale)

        if self.training:
            # if training phase, then sample RoIs
            sample_roi, gt_roi_loc, gt_roi_label = self.sample_proposal_module(roi,
                                                                               at.tonumpy(gt_bbox),
                                                                               at.tonumpy(gt_label),
                                                                               self.loc_normalize_mean,
                                                                               self.loc_normalize_std)

            # get location of ground-truth bounding boxes
            gt_rpn_loc, gt_rpn_label = self.sample_anchor_module(at.tonumpy(gt_bbox),
                                                                 anchors,
                                                                 img_size)
            gt_rpn_loc = at.totensor(gt_rpn_loc)
            gt_rpn_label = at.totensor(gt_rpn_label).long()

            # bounding-box regression loss
            rpn_loc_loss = bbox_regression_loss(loc,
                                                gt_rpn_loc,
                                                gt_rpn_label.data,
                                                self.rpn_sigma)

            # foreground-background classification loss
            rpn_cls_loss = F.cross_entropy(score, gt_rpn_label.cuda(), ignore_index=-1)

            return sample_roi, gt_roi_loc, gt_roi_label, rpn_loc_loss, rpn_cls_loss

        return roi

class FPNBasedRPN(RegionProposalNetwork):
    def __init__(self, scales, ratios, rpn_conv, rpn_loc, rpn_score):
        super(FPNBasedRPN, self).__init__(scales, ratios)
        self.n_anchor = len(ratios)
        self.rpn_conv = rpn_conv
        self.rpn_loc = rpn_loc
        self.rpn_score = rpn_score
        normal_init(self.rpn_conv, 0, 0.01)
        normal_init(self.rpn_loc, 0, 0.01)
        normal_init(self.rpn_score, 0, 0.01)

    def forward(self, feature_maps, img_size, scale, gt_bbox, gt_label):
        n = 1  # batch size is always one
        feature_shapes = []
        locs, scores, fg_scores = [], [], []

        # parse all feature maps
        for feature in feature_maps:
            h = F.relu(self.rpn_conv(feature))

            loc = self.rpn_loc(h)   # (x,y,h,w)
            score = self.rpn_score(h)

            h, w = loc.shape[2:]

            # get bbox location and score
            loc = loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
            score = score.permute(0, 2, 3, 1).contiguous()

            # softmax score
            softmax_score = F.softmax(score.view(n, h, w, self.n_anchor, 2), dim=4)
            fg_score = softmax_score[:, :, :, :, 1].contiguous().view(n, -1)

            feature_shapes.append((h, w))
            locs.append(loc)
            scores.append(score.view(n, -1, 2))
            fg_scores.append(fg_score)

        loc = t.cat(locs, dim=1)[0]
        score = t.cat(scores, dim=1)[0]
        fg_score = t.cat(fg_scores, dim=1)[0]

        # get feature stride
        feat_strides = []
        for shape in feature_shapes:
            feat_strides.append(img_size[0] // shape[0])
        # NOTE: generate anchors (all layers)
        anchors = generate_anchors_fpn(self.scales, self.ratios, feature_shapes, feat_strides)

        # get proposals given anchors and bbox offsets
        rois = self.generate_proposals_module(loc.cpu().data.numpy(),
                                             fg_score.cpu().data.numpy(),
                                             anchors,
                                             img_size,
                                             scale)

        if self.training:
            # if training phase, then sample RoIs
            sample_roi, gt_roi_loc, gt_roi_label = self.sample_proposal_module(rois,
                                                                               at.tonumpy(gt_bbox),
                                                                               at.tonumpy(gt_label),
                                                                               self.loc_normalize_mean,
                                                                               self.loc_normalize_std)

            # get location of ground-truth bounding boxes
            gt_rpn_loc, gt_rpn_label = self.sample_anchor_module(at.tonumpy(gt_bbox),
                                                                 anchors,
                                                                 img_size)
            gt_rpn_loc = at.totensor(gt_rpn_loc)
            gt_rpn_label = at.totensor(gt_rpn_label).long()

            # bounding-box regression loss
            rpn_loc_loss = bbox_regression_loss(loc,
                                                gt_rpn_loc,
                                                gt_rpn_label.data,
                                                self.rpn_sigma)

            # foreground-background classification loss
            rpn_cls_loss = F.cross_entropy(score, gt_rpn_label.cuda(), ignore_index=-1)

            return sample_roi, gt_roi_loc, gt_roi_label, rpn_loc_loss, rpn_cls_loss

        return rois


