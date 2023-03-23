import csv
import os

from PIL import Image

from torch.utils.data import Dataset


class Kitti(Dataset):
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
    image_dir_name = "image_2"
    labels_dir_name = "label_2"

    def __init__(self, train=True, transforms=None):
        self.images = []
        self.labels = []
        self.root = os.getcwd()     # absolute path? relative path?
        self.train = train
        self.transforms = transforms
        self.sub_set = "training" if self.train else "testing"
        self.dataset_dir = os.path.join(self.root,self.__class__.__name__,self.sub_set)
        
        if self.train:  
            # training set
            image_dir = os.path.join(self.dataset_dir, self.image_dir_name)
            labels_dir = os.path.join(self.dataset_dir, self.labels_dir_name)
            for img_file in os.listdir(image_dir):
                self.images.append(os.path.join(image_dir, img_file))
                self.labels.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))
        else:           
            # testing set
            image_dir = os.path.join(self.dataset_dir, self.image_dir_name)
            for img_file in os.listdir(image_dir):
                self.images.append(os.path.join(image_dir, img_file))

    def __getitem__(self, i):
        """Get item at a given index.

        Args:
            i (int): Index
        Returns:
            tuple: (image, params) where 'params' (dict) with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float
        """
        image = Image.open(self.images[i])

        if self.train:
            params = self.parse_label(i)
        else: 
            params = None

        if self.transforms:
            image, params = self.transforms(image, params)
        return image, params

    def parse_label(self, index):
        params = []
        with open(self.labels[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                params.append(
                    {
                        "type": line[0],
                        "truncated": float(line[1]),
                        "occluded": int(line[2]),
                        "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                        "rotation_y": float(line[14]),
                    }
                )
        return params

    def __len__(self):
        return len(self.images)
