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
from data.dataset import inverse_normalize,TestDataset,Dataset
# from data.voc_dataset import VOCBboxDataset
#from data.kitti_dataset import KITTIDataset

# model 
from model import FasterRCNNVGG16
from torchnet.meter import AverageValueMeter
from model.faster_rcnn_vgg16 import LossTuple

# utils
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_voc


def update_meters(meters, losses):
    loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
    for key, meter in meters.items():
        meter.add(loss_d[key])

def reset_meters(meters):
    for _, meter in meters.items():
        meter.reset()


def get_meter_data(meters):
    return {k: v.value()[0] for k, v in meters.items()}

def save_model(model, model_name, epoch):
    save_path = f'./checkpoints/{model_name}/{epoch}.pth'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_path)

def build_optimizer(net):
    """
    return optimizer, It could be overwriten if you want to specify 
    special optimizer
    """
    lr = opt.lr
    params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
    if opt.use_adam:
        return torch.optim.Adam(params)
    else:
        return torch.optim.SGD(params, momentum=0.9)

def train(**kwargs):

    # set up cuda
    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

    # parse model parameters from config 
    opt.f_parse_args(kwargs)

    # load training dataset 
    dataset = Dataset(opt)

    # img, bbox, label, scale = dataset[1]
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
    
    print('data completed')

    # model construction 
    net = FasterRCNNVGG16().to(device)
    print('model completed')

    # optimizer construction
    optimizer = build_optimizer(net)
    print('optimizer completed')

    # fitting 
    meters = {k: AverageValueMeter() for k in LossTuple._fields}

    best_mAP = 0
    lr = opt.lr
    for epoch in range(1, opt.epoch + 1):
        # switch to train mode
        net.train()
        # reset meters
        reset_meters(meters)
        # train batch
        for img, bbox, label, scale in tqdm(dataloader):
            # prepare data
            scale = at.scalar(scale)
            img, bbox, label = img.to(device).float(), bbox.to(device), label.to(device)

            # forward + backward
            optimizer.zero_grad()
            losses = net.forward(img, bbox, label, scale)
            losses.total_loss.backward()
            optimizer.step()
            update_meters(meters, losses)
        
        # print loss
        print('learning rate: ', lr)
        loss_metadata = get_meter_data(meters)
        rpn_loc_loss = loss_metadata['rpn_loc_loss']
        rpn_cls_loss = loss_metadata['rpn_cls_loss']
        roi_loc_loss = loss_metadata['roi_loc_loss']
        roi_cls_loss = loss_metadata['roi_cls_loss']
        print('epoch:{}, rpn_loc_loss:{}, rpn_cls_loss:{}, roi_loc_loss:{}, roi_cls_loss:{}'.format(epoch, rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss))
        print(f'total_loss: {loss_metadata["total_loss"]:.4f}')
        
        # evaluate
        net.eval()
        mAP = eval_voc(test_dataloader, net)

        # save model (if best model)
        if mAP > best_mAP:
            best_mAP = mAP
            best_path = save_model(best_mAP, opt.model, epoch)
        
        # learning rate decay
        if epoch == opt.epoch_decay:
            # load best model
            state_dict = torch.load(best_path)
            net.load_state_dict(state_dict)
            # learning rate decay
            for param in optimizer.param_groups:
                param['lr'] *= opt.lr_decay
            lr = lr * opt.lr_decay
    
    # save final model
    PATH = f'{opt.save_dir}/fasterrcnn_vgg16.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    train()