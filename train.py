from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm
import torch

# config
from utils.config import opt

#dataset
from torch.utils.data import DataLoader
from data.dataset import inverse_normalize,TestDataset
from data.voc_dataset import VOCBboxDataset
from data.kitti_dataset import KITTIDataset

# model 
from model import FasterRCNNVGG16

# utils
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):

    # set up cuda
    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

    # parse model parameters from config 
    opt.f_parse_args(kwargs)

    # load training dataset 
    dataset = VOCBboxDataset(opt)

    img, bbox, label, scale = dataset[1]
    # print('bbox type: ',type(bbox))
    # print('bbox: ',bbox)
    # print('label: ', label)
    # print('label type: ', type(label))
    # print('scale: ',scale)
    # print('load data')
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=True,
                            num_workers=opt.num_workers,
                            pin_memory=True)
        
    # # load testing dataset
    testset = TestDataset(opt)
    test_dataloader = DataLoader(testset,
                                 batch_size=1,
                                 shuffle=False, 
                                 num_workers=opt.test_num_workers,
                                 pin_memory=True)
    
    # construct model
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).to(device)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr = opt.lr

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for idx, (img, bbox, label, scale) in tqdm(enumerate(dataloader)):
            # send data to device
            scale = at.scalar(scale)
            img, bbox, label = img.to(device).float(), bbox.to(device), label.to(device)

            # train one batch
            trainer.train_step(img, bbox, label, scale)
            
            # visualization
            if opt.visualize:
                if (idx + 1) % opt.plot_every == 0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

                    # plot loss
                    trainer.vis.plot_many(trainer.get_meter_data())

                    # plot groud truth bboxes
                    ori_img = inverse_normalize(at.tonumpy(img[0]))
                    gt_img = visdom_bbox(ori_img,
                                        at.tonumpy(bbox[0]),
                                        at.tonumpy(label[0]))
                    trainer.vis.img('gt_img', gt_img)

                    # plot predict bboxes
                    pred_bboxes, pred_labels, pred_scores = trainer.faster_rcnn.predict([ori_img], visualize=True)
                    pred_img = visdom_bbox(ori_img,
                                        at.tonumpy(pred_bboxes[0]),
                                        at.tonumpy(pred_labels[0]).reshape(-1),
                                        at.tonumpy(pred_scores[0]))
                    trainer.vis.img('pred_img', pred_img)

                    # rpn confusion matrix(meter)
                    trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                    # roi confusion matrix
                    trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        
        # evaluate
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        print('lr:{}, mAP:{},loss:{}'.format(str(lr),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data())))

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr *= opt.lr_decay

        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire
    
    fire.Fire()