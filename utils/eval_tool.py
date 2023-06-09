from __future__ import division

from collections import defaultdict
import itertools
import numpy as np
import six
import torch
from tqdm import tqdm

from model.utils.bbox_tools import bbox_iou
from utils import array_tool as at

def run_test(net, test_loader):
    # Evaluation for VOC dataset
    pred_bboxes, pred_labels, pred_scores = [], [], []
    gt_bboxes, gt_labels, gt_difficults = [], [], []
    with torch.no_grad():
        for img, bbox, label, scale, ori_size, difficult in tqdm(test_loader):
            scale = at.scalar(scale)
            original_size = [ori_size[0][0].item(), ori_size[1][0].item()]
            pred_bbox, pred_label, pred_score = net(img, None, None, scale,original_size)
            gt_bboxes += list(bbox.numpy())
            gt_labels += list(label.numpy())
            gt_difficults += list(difficult.numpy())
            pred_bboxes += [pred_bbox]
            pred_labels += [pred_label]
            pred_scores += [pred_score]
    
    return pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults


def voc_ap(net, test_loader):
    # Evaluation for VOC dataset

    # initialize evaluation metrics
    map_result = {'mAP': 0, 'mAP_0.5': 0, 'mAP_0.75': 0}
    iou_threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # run test on test dataset
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults = run_test(net, test_loader)

    # evaluate results regardless of area size
    for iou_thresh in iou_threshes:
        result = eval_voc(pred_bboxes, pred_labels, pred_scores,
                          gt_bboxes, gt_labels, gt_difficults,
                          iou_thresh, use_07_metric=True)
        # accumulate results
        map_result['mAP'] += result['map']
        # save map for iou 0.5 & 0.75
        if iou_thresh == 0.5:
            map_result['mAP_0.5'] = result['map']
        elif iou_thresh == 0.75:
            map_result['mAP_0.75'] = result['map']
    
    map_result['mAP'] /= 10

    # print results
    print('Result: mAP=={:.2f} | mAP@0.5== {:.2f} | mAP@0.75=={:.2f}'.format(map_result['mAP']*100, map_result['mAP_0.5']*100, map_result['mAP_0.75']*100))


    return map_result['mAP']


def eval_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC."""

    precision, recall = voc_prec_rec(pred_bboxes, 
                                          pred_labels, 
                                          pred_scores,
                                          gt_bboxes, 
                                          gt_labels, 
                                          gt_difficults,
                                          iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(precision, recall, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    precision = [None] * n_fg_class
    recall = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        precision[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            recall[l] = tp / n_pos[l]

    return precision, recall


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


