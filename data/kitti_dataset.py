import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image



import cv2
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer


# helper functions for converting KITTI to VOC format



class KITTIDataset:
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            Dataset dependency::

                  data
                    └── Kitti
                        ├── training
                        |   ├── image_2
                        |   └── label_2
                        └── testing
                            └── image_2
        train (bool): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.


    """
    def __init__(self, data_dir, split, use_difficult=False):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.label_names = KITTI_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)
    
    def get_sample(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.
        Args:
            i (int): The index of the example.
        Returns:
            tuple of an image and bounding boxes
        """
        id_ = self.ids[i]
        # parse annotation
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(KITTI_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult
    
    __getitem__= get_sample

KITTI_BBOX_LABEL_NAMES = (
    'car', 
    'pedestrian', 
    'cyclist')
