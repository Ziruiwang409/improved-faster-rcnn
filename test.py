import torch

# config
from utils.config import opt

#dataset
from torch.utils.data import DataLoader
from data.dataset import TestDataset
# from data.voc_dataset import VOCBboxDataset
# from data.kitti_dataset import KITTIDataset

# model 
from model import FasterRCNNVGG16

# utils
from utils.eval_tool import evaluate

def test(**kwargs):

    # set up cuda
    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

    # parse model parameters from config 
    opt.f_parse_args(kwargs)
    # # load testing dataset
    testset = TestDataset(opt)
    test_dataloader = DataLoader(testset,
                                 batch_size=1,
                                 shuffle=False, 
                                 num_workers=opt.test_num_workers)
    
    print('data completed')

    # model construction 
    net = FasterRCNNVGG16().to(device)
    print('model completed')

    # load pretrained weight
    PATH = f'{opt.save_dir}/fasterrcnn_vgg16.pth'
    net.load_state_dict(torch.load(PATH))
    print('load weight completed')

    # evaluation
    net.eval()
    aps = evaluate(net, test_dataloader, device=device, dataset=testset)



if __name__ == '__main__':
    test()