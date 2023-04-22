import torch

# config
from utils.config import opt

#dataset
from torch.utils.data import DataLoader
from data.dataset import Dataset
# from data.voc_dataset import VOCBboxDataset
# from data.kitti_dataset import KITTIDataset

# model 
from model import FasterRCNNVGG16

# utils
from utils.eval_tool import voc_ap

def test(**kwargs):

    # set up cuda
    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

    # parse model parameters from config 
    opt.f_parse_args(kwargs)
    # # load testing dataset
    testset = Dataset(opt, mode='test')
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
    print('pretrained weight loaded')

    # evaluation
    net.eval()
    



if __name__ == '__main__':
    test()