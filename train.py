from __future__ import  absolute_import
import os
import sys

from tqdm import tqdm
import torch

# config
from utils.config import opt

#dataset
from torch.utils.data import DataLoader
from data.dataset import inverse_normalize
from data.dataset import Dataset
# from data.voc_dataset import VOCBboxDataset
#from data.kitti_dataset import KITTIDataset

# model 
from model import FasterRCNNVGG16, FPNFasterRCNNVGG16
from torchnet.meter import AverageValueMeter
from model.frcnn_bottleneck import Losses

# utils
from utils import array_tool as at
from utils.eval_tool import voc_ap

def setup_logger():
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    f = open(f'{opt.save_dir}/log.txt', 'w')
    sys.stdout = f

    return f


def update_meters(meters, losses):
    loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
    for key, meter in meters.items():
        meter.add(loss_d[key])

def reset_meters(meters):
    for _, meter in meters.items():
        meter.reset()


def get_meter_data(meters):
    return {k: v.value()[0] for k, v in meters.items()}

def save_model(model, model_name, dataset_name,epoch):
    PATH = f'./checkpoints/{model_name}/{dataset_name}/checkpoint{epoch}.pth'
    dir = os.path.dirname(PATH)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(), PATH)

    return PATH

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
    
    return torch.optim.SGD(params, momentum=0.9)


def train(**kwargs):

    # set up cuda
    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')


    # parse model parameters from config 
    opt.f_parse_args(kwargs)

    print('Load dataset')
    # load training dataset 
    train_data = Dataset(opt,mode='train')
    train_dataloader = DataLoader(train_data, 
                            batch_size=1, 
                            shuffle=True,
                            num_workers=opt.train_num_workers)
        
    # # load testing dataset
    test_data = Dataset(opt, mode='test')
    test_dataloader = DataLoader(test_data,
                                 batch_size=1,
                                 shuffle=False, 
                                 num_workers=opt.test_num_workers)
    

    print('Load model')
    # model construction
    if opt.database == 'voc': 
        if opt.apply_fpn:
            net = FPNFasterRCNNVGG16(n_fg_class=20).to(device)
        else:
            net = FasterRCNNVGG16(n_fg_class=20).to(device) 
    elif opt.database == 'kitti':
        if opt.apply_fpn:
            net = FPNFasterRCNNVGG16(n_fg_class=3).to(device)
        else:
            net = FasterRCNNVGG16(n_fg_class=3).to(device)  # 3 classes: Car, Pedestrian, Cyclist

    print('Load optimizer')
    # optimizer construction
    optimizer = build_optimizer(net)

    print('Load hyperparameters')
    # fitting 
    meters = {k: AverageValueMeter() for k in Losses._fields}
    best_mAP = 0
    best_path = None
    lr = opt.lr


    print('Start training...')
    for epoch in range(1, opt.epoch + 1):
        # switch to train mode
        net.train()
        # reset meters
        reset_meters(meters)
        # train batch
        for img, bboxes, labels, scale in tqdm(train_dataloader):
            # prepare data
            scale = at.scalar(scale)
            img, bboxes, labels = img.to(device).float(), bboxes.to(device), labels.to(device)
            # forward + backward
            optimizer.zero_grad()
            losses = net.forward(img,bboxes, labels, scale)
            losses.total_loss.backward()
            optimizer.step()
            update_meters(meters, losses)
        
        # print loss
        loss_metadata = get_meter_data(meters)
        rpn_loc_loss = loss_metadata['rpn_loc_loss']
        rpn_cls_loss = loss_metadata['rpn_cls_loss']
        roi_loc_loss = loss_metadata['roi_loc_loss']
        roi_cls_loss = loss_metadata['roi_cls_loss']
        total_loss = loss_metadata['total_loss']
        print('epoch #{}: lr=={} | rpn_loc_loss=={:.4f} | rpn_cls_loss=={:.4f} | roi_loc_loss=={:.4f} | roi_cls_loss=={:.4f} | total_loss=={:.4f}'.format(epoch,
                                                                                                                                                          lr, 
                                                                                                                                                          rpn_loc_loss, 
                                                                                                                                                          rpn_cls_loss, 
                                                                                                                                                          roi_loc_loss, 
                                                                                                                                                          roi_cls_loss,
                                                                                                                                                          total_loss))

        # evaluate
        net.eval()
        

        mAP = voc_ap(net, test_dataloader)


        # save model (if best model)
        if mAP > best_mAP:
            best_mAP = mAP
            best_path = save_model(net, opt.model,opt.database, epoch)
        
        # learning rate decay
        if epoch == opt.epoch_decay:
            # load best model
            net.load_state_dict(torch.load(best_path))
            # learning rate decay
            for param in optimizer.param_groups:
                param['lr'] *= opt.lr_decay
            lr = lr * opt.lr_decay
    
    # save final model
    model_name = 'frcnn_vgg16' if not opt.apply_fpn else 'fpn_frcnn_vgg16'
    PATH = f'{opt.save_dir}/{opt.database}/{model_name}.pth'
    target_dir = os.path.dirname(PATH)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    torch.save(net.state_dict(), PATH)

    #log.close()


if __name__ == '__main__':
    train()