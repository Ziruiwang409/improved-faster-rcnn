import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable


def decoder(pred):
    """
    pred (tensor) 1xSxSx(B*5+C)  -- in our case with resnet: 1x14x14x(2*5+20)
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    """
    grid_num = pred.squeeze().shape[0]  # 14 for resnet50 base, 7 for vgg16
    assert pred.squeeze().shape[0] == pred.squeeze().shape[1]  # square grid
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1.0 / grid_num
    pred = pred.data
    pred = pred.squeeze(0)  # SxSx(B*5+C)
    object_confidence1 = pred[:, :, 4].unsqueeze(2)
    object_confidence2 = pred[:, :, 9].unsqueeze(2)
    object_confidences = torch.cat((object_confidence1, object_confidence2), 2)

    # Select all predictions above the threshold
    min_confidence_threshold = 0.1
    mask1 = object_confidences > min_confidence_threshold

    # We always want to select at least one predictions so we also take the prediction with max confidence
    mask2 = object_confidences == object_confidences.max()
    mask = (mask1 + mask2).gt(0)

    # We need to convert the grid-relative coordinates back to image coordinates
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = pred[i, j, b * 5 : b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = (
                        torch.FloatTensor([j, i]) * cell_size
                    )  # upper left corner of grid cell
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(
                        box.size()
                    )  # convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob * max_prob)
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)
        probs = torch.cat(probs, 0)
        cls_indexs = torch.stack(cls_indexs, dim=0)

    # Perform non-maximum suppression so that we don't predict many similar and overlapping boxes
    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


def nms(bboxes, scores, threshold=0.5):
    """
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0] if order.numel() > 1 else order.item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


def predict_image(model, image_name, root_img_directory=""):
    """
    Predict output for a single image
    :param model: detector model for inference
    :param image_name: image file name e.g. '0000000.jpg'
    :param root_img_directory:
    :return: List of lists containing:
        - (x1, y1)
        - (x2, y2)
        - predicted class name
        - image name
        - predicted class probability
    """

    result = []
    image = cv2.imread(os.path.join(root_img_directory + image_name))
    h, w, _ = image.shape
    img = cv2.resize(image, (YOLO_IMG_DIM, YOLO_IMG_DIM))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = VOC_IMG_MEAN
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    img = transform(img)
    with torch.no_grad():
        img = Variable(img[None, :, :, :])
        img = img.cuda()

        pred = model(img)  # 1xSxSx(B*5+C)
        pred = pred.cpu()
        boxes, cls_indexs, probs = decoder(pred)

        for i, box in enumerate(boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)
            cls_index = cls_indexs[i]
            cls_index = int(cls_index)  # convert LongTensor to int
            prob = probs[i]
            prob = float(prob)
            result.append(
                [(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob]
            )
    return result

VOC_CLASSES = (  # always index 0
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

VOC_IMG_MEAN = (123, 117, 104)  # RGB

COLORS = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

# network expects a square input of this dimension
YOLO_IMG_DIM = 448