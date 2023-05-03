from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from data.kitti_dataset import KITTIDataset
from skimage import transform as sktsf
import torchvision.transforms.functional as F
from data.util import *
import numpy as np
from utils.config import opt




def normalize(img):
    img = F.normalize(
        tensor=t.from_numpy(img),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True
    ).numpy()
    return img



def resize(img, min_size=600, max_size=1000):
    """Resize and normalize an image for feature extraction.
    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.
    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.
    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    return img


class Transform(object):

    def __init__(self, min_size=600, max_size=1000, mode='train'):
        self.min_size = min_size
        self.max_size = max_size
        self.mode = mode

    def __call__(self, input_data):
        # get original image size
        img, bbox, label, difficult = input_data
        ori_img = np.copy(img)
        ori_size = ori_img.shape[1:]
        # resize image
        trans_img = resize(ori_img, self.min_size, self.max_size)
        trans_size = trans_img.shape[1:]
        # get scale
        trans_img = normalize(trans_img)
        scale = trans_size[0] / ori_size[0]

        # Transform if training
        if self.mode == 'train':
            # resize bbox
            bbox = resize_bbox(bbox, ori_size, trans_size)
            # image transformation
            trans_img, params = random_flip(trans_img, x_random=True, return_param=True)
            # bbox transformation
            bbox = flip_bbox(bbox, trans_size, x_flip=params['x_flip'])

            return trans_img.copy(), bbox.copy(), label.copy(), scale
        elif self.mode == 'test':
            return trans_img, bbox, label, scale, ori_size, difficult
        elif self.mode == 'vis':
            return ori_img, trans_img, scale, ori_size

class Dataset:
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode
        if opt.database == 'voc':
            if self.mode == 'train':
                self.db = VOCBboxDataset(opt.voc_data_dir, split='trainval')
            else:
                self.db = VOCBboxDataset(opt.voc_data_dir, split='test', use_difficult=True)
        elif opt.database == 'kitti':
            if self.mode == 'train':
                self.db = KITTIDataset(opt.kitti_data_dir, split='train')
            else:
                self.db = KITTIDataset(opt.kitti_data_dir, split='val')
        self.tsf = Transform(opt.min_size, opt.max_size, mode=mode)

    def __getitem__(self, idx):
        input_data = self.db.get_sample(idx)
        return self.tsf(input_data)

    def __len__(self):
        return len(self.db)


