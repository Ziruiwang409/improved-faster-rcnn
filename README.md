# Object Detector for Autonomous Vehicles Based on Faster RCNN 
Zirui Wang

- Pytorch implementation of Faster R-CNN based on VGG16.
- Supports Feature Pyramid Network (FPN).
- Supports Deformable Convolution (DCNv1) 
- Pretrained model can be found [here](https://drive.google.com/drive/folders/1XCTZjdumgNVPtH3OI-FrcWsyC0-uYHGG?usp=sharing)

## 1. Introduction

![architecture](./images/model_architecture.png)

This is a implementation a framework that combines Feature Pyramid Network (FPN) and Deformable 
Convolution Network (DCNv1) to improve Faster RCNN on object detection tasks.The whole model is 
implemented on Pytorch and trained on VOC 2007 training set and evaluate on VOC 2007 test set, 
with 1.1% improvement on mAP@[.5,.95] score and 3.95% improvement on mAP@[0.75:0.95] score, which 
demonstrates the effectiveness of the model. The model also support for KITTI 2d Object Detection 
dataset, training on KITTI 2D object detection training set and evaluate on validation set, with a 
surprisingly 11.96% increase on mAP@[.5,.95] score and a 23.35% increase on mAP@[.75,.95]. m

## 2. Experimental Results

- Detection results on PASCAL VOC 2007 test set
  - All models were evaluated using COCO-style detection evaluation metrics.

| Training dataset |        Model         |   mAP@[.5,.95]  |   mAP@[.75,.95]  |
| :--------------: | :------------------: | :-------------: | :--------------: |
|      VOC 07      |    Faster RCNN       |      69.65      |      31.14       | 
|      VOC 07      |   FPN+ Faster RCNN   |      69.83      |    **34.02**     |
|      VOC 07      |  Deform+ Faster RCNN |    **69.93**    |      30.85       |  

- Detection results on KITTI 2d Object Detection valication set
  - All models were evaluated using COCO-style detection evaluation metrics.

| Training dataset |          Model           |   mAP@[.5,.95]  |   mAP@[.75,.95]  |
| :--------------: | :----------------------: | :-------------: | :--------------: |
|     KITTI 2d     |        Faster RCNN       |      71.58      |      32.40       | 
|     KITTI 2d     |      FPN+ Faster RCNN    |    **82.76**    |      56.02       | 
|     KITTI 2d     |    Deform+ Faster RCNN   |      71.73      |      33.16       |  
|     KITTI 2d     | FPN+ Deform+ Faster RCNN |      82.59      |    **56.30**     |

## 3. Requirements

- numpy
- six
- torch
- torchvision
- tqdm
- cv2
- defaultdict
- itertools
- namedtuple
- skimage
- xml
- pascal_voc_writer
- PIL

## 4. Usage

### 4.1 Data preparation

- Download the training, validation, and test data.

```shell
# VOC 2007 trainval and test datasets
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# KITTI 2d Object Detection training set and groundtruth labels
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
```

- Untar files into two separate directories named `VOCdevkit` and `KITTIdevkit`

```shell
# VOC 2007 trainval and test datasets
mkdir VOCdevkit && cd VOCdevkit
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar

# KITTI 2d Object Detection trainset and labels (following last command)
cd ..
mkdir KITTIdevkit && cd KITTI devkit
unzip data_object_image_2.zip
unzip data_object_label_2.zip
```

- The KITTI dataset need to reformat to match with the following structure. 

```shell
dataset
   ├── KITTIdevkit
   │   ├── training
   │       ├── image_2
   │       └── label_2
   └── VOCdevkit
       ├── VOC2007
           ├── Annotations
           ├── ImageSets
           ├── JPEGImages
           ├── SegmentationClass
           └── SegmentationObject
```

- Convert the KITTI dataset into PASCAL VOC 2007 dataset format using the dataset format convertion tool script

```shell
# go back to the main page of the project code
cd ./improved_faster_rcnn
# change directory to find the format convertion script
cd data
# run dataset format convertion script
python kitti2voc.py
```

- After running the above command, you should have the same dataset structure for KITTI as VOC 2007, and it is now 
ready to load into the model

```shell
dataset
   ├── KITTI2VOC
   │   ├── Annotations
   │   ├── ImageSets
   │   ├── JPEGImages
   └── VOCdevkit
       └── VOC2007
           ├── Annotations
           ├── ImageSets
           ├── JPEGImages
           ├── SegmentationClass
           └── SegmentationObject
```

### 4.2 Train models

- You can easily modify the parameters for training in `utils/config.py` and run the following script for model training

```shell
# if you are in local environemnt, run:
python3 ./train.py
# if you are in conda environment, run: 
python ./train.py
```

### 4.3 Test models

- You can easily modify the parameters for testing in `utils/config.py` and run the following script for model training
- You can visualize the testing image by setting `visualize=True` in configuration file
- The output image should be placed under `save_dir/visuals` specified in the configuration file

- Ex 1) FPN based on VGG16 (file name: "fpn_vgg16_1.pth")

```shell
# if you are in local environemnt, run:
python3 ./test.py
# if you are in conda environment, run: 
python ./test.py
```

### 4.4 Example output images

- Below are some of the resulting images from visualization

<p align="center">
  <img src="./images/output3.jpg">
</p>
<p align="center">
  <img src="./images/output6.jpg">
</p>
<p align="center">
  <img src="./images/output8.jpg">
</p>

## 5. Reference

- [simple-faster-rcnn](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
- [fpn.pytorch](https://github.com/jwyang/fpn.pytorch)
- [Faster-RCNN-FPN](https://github.com/txytju/Faster-RCNN-FPN)
- [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets)
- [PyTorch-Deformable-Convolution-v2](https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2)
