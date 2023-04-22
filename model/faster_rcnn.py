
# Pytorch packages
import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

# Python packages
import numpy as np
from collections import namedtuple

# Util packages
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from utils.config import opt

# Loss Tuple
Losses = namedtuple('Losses',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their 
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in 
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that 
        belong to the proposed RoIs, classify the categories of the objects 
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. 
    Faster R-CNN: Towards Real-Time Object Detection with 
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(self, extractor, rpn, predictor, n_classes, loc, score,
                 spatial_scale, pooling_size, roi_sigma):
        super(FasterRCNN, self).__init__()
        # architecture parameters
        self.extractor = extractor
        self.rpn = rpn
        self.predictor = predictor

        # hyper parameters
        self.n_classes = n_classes
        self.loc = loc
        self.score = score

        # hyper parameters for roi pooling
        self.spatial_scale = spatial_scale
        self.pooling_size = pooling_size
        self.roi_sigma = roi_sigma

        # hyper parameters for evaluation
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.nms_thresh = opt.nms_thresh        # threshold for non maximum suppression on bbox
        self.score_thresh = opt.score_thresh    # threshold for bbox score
        

    def forward(self, x, gt_bboxes, gt_labels, scale, ori_size=None):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        # train
        if self.training:             # (nn.Module parameters training)
            img_size = tuple(x.shape[2:])
            # feature extraction (Backbone CNN: VGG16)
            feature = self.feature_extraction_layer(x)    
            # print("gt_bboxes:", gt_bboxes)
            # print("gt_labels:", gt_labels)
            # RPN (NOTE: FPN based Region Proposal Network)
            roi, gt_roi_loc, gt_roi_label, rpn_loc_loss, rpn_cls_loss = self.rpn(feature, img_size, scale, gt_bboxes[0], gt_labels[0])

            # RoI pooling 
            roi_pool_feature = self.roi_pooling_layer(feature, roi)

            # Bounding Box regression and classification
            roi_loc, roi_score = self.bbox_regression_and_classification_layer(roi_pool_feature)

            # Calculate losses
            n_sample = roi_loc.shape[0]
            roi_loc = roi_loc.view(n_sample, -1, 4)
            roi_loc = roi_loc[t.arange(0, n_sample).long().cuda(),
                              at.totensor(gt_roi_label).long()]

            gt_roi_loc = at.totensor(gt_roi_loc)
            gt_roi_label = at.totensor(gt_roi_label).long()

            roi_loc_loss = bbox_regression_loss(roi_loc.contiguous(),
                                                gt_roi_loc,
                                                gt_roi_label.data,
                                                self.roi_sigma)

            roi_cls_loss = F.cross_entropy(roi_score, gt_roi_label.cuda())

            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
            losses = losses + [sum(losses)]

            return Losses(*losses)
        else:   # test
            x = at.totensor(x).float()
            img_size = tuple(x.shape[2:])

            # feature extraction (Backbone CNN: VGG16)
            feature = self.feature_extraction_layer(x)

            # RPN (NOTE: FPN based Region Proposal Network)
            roi= self.rpn(feature, img_size, scale, None, None)

            # RoI pooling 
            roi_pool_feature = self.roi_pooling_layer(feature, roi)

            # Bounding Box regression and classification
            roi_cls_loc, roi_score = self.bbox_regression_and_classification_layer(roi_pool_feature)

            # We are assuming that batch size is 1.
            roi_score = roi_score.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(roi) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_classes)[None]
            std = t.Tensor(self.loc_normalize_std).cuda().repeat(self.n_classes)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_classes, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_classes * 4)

            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=ori_size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=ori_size[1])

            prob = (F.softmax(at.totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)

            return bbox, label, score


    # NOTE: Override in the child class (improved_faster_rcnn.py)
    def feature_extraction_layer(self, x):
        raise NotImplementedError

    # NOTE: Override in the child class (improved_faster_rcnn.py)
    def roi_pooling_layer(self, feature, roi):
        raise NotImplementedError

    # NOTE: Override in the child class (improved_faster_rcnn.py)
    def bbox_regression_and_classification_layer(self, roi_pool_feat):
        raise NotImplementedError

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_classes):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_classes
            , 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l,self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    
def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def bbox_regression_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss




