from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:

    # dataset params
    database = 'VOC'  # choose one from ['VOC', 'KITTI', 'COCO']
    voc_data_dir = '/data/ziruiw3/VOCdevkit/VOC2007/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    train_num_workers = 8
    test_num_workers = 8

    nms_thresh = 0.3        # iou threshold in nms
    score_thresh = 0.05     # score threshold in nms

    # sigma for l1_smooth_loss
    rpn_sigma = 3.          
    roi_sigma = 1.          

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr = 1e-3
    lr_decay = 0.1


    # preset
    model = 'vgg16'  # choose one from ['vgg16', 'resnet50']
    backbone = 'vgg16' # choose one from ['vgg16', 'resnet50']

    # training
    epoch = 15
    epoch_decay = 10

    test_num = 10000
    save_dir = './outputs'

    def f_parse_args(self, kwargs):
        '''
            Helper function that parse user input pamameter
        '''
        params = {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k == 'f_parse_args'}
        # parse user input argument
        for key, value in kwargs.items():
            if key not in params:
                raise ValueError('UnKnown Option: "--%s"' % key)
            setattr(self, key, value)

        print('=================== User config ===============')
        pprint(params)
        print('===============================================')

opt = Config()
