from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
import torchvision.transforms.functional as F
from data.util import *
import numpy as np
from utils.config import opt


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def normalze(img):
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

    def __init__(self, min_size=600, max_size=1000, train=True):
        self.min_size = min_size
        self.max_size = max_size
        self.train = train

    def __call__(self, input_data):
        # get original image size
        img, bbox, label, difficult = input_data
        ori_size = img.shape[1:]
        # resize image
        img = resize(img, self.min_size, self.max_size)
        trans_size = img.shape[1:]
        # get scale
        img = normalze(img)
        scale = trans_size[0] / ori_size[0]

        # Transform if training
        if self.train:
            # resize bbox
            bbox = resize_bbox(bbox, ori_size, trans_size)
            # image transformation
            img, params = random_flip(img, x_random=True, return_param=True)
            # bbox transformation
            bbox = flip_bbox(bbox, trans_size, x_flip=params['x_flip'])

            return img.copy(), bbox.copy(), label.copy(), scale
        else:
            return img, bbox, label, scale, ori_size, difficult

class Dataset:
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.train = True if mode == 'train' else False
        if opt.database == 'VOC':
            if self.train:
                self.db = VOCBboxDataset(opt.voc_data_dir, split='trainval')
            else:
                self.db = VOCBboxDataset(opt.voc_data_dir, split='test', use_difficult=True)
        elif opt.database == 'KITTI':
            if self.train:
                self.db = KITTIDataset(opt.kitti_data_dir, split='train')
            else:
                self.db = KITTIDataset(opt.kitti_data_dir, split='test')
        self.tsf = Transform(opt.min_size, opt.max_size, train=self.train)

    def __getitem__(self, idx):
        input_data = self.db.get_sample(idx)
        return self.tsf(input_data)

    def __len__(self):
        return len(self.db)


