from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    dataset_choice = 'voc'  # choose one from 'voc', 'kitti', 'coco'
    voc_data_dir = '/data/ziruiw3/VOCdevkit/VOC2007/'
    kitti_data_dir = '/data/ziruiw3/VOCdevkit/VOC2007/'         # TODO: change path to kitti dataset   
    coco_data_dir = '/data/ziruiw3/VOCdevkit/VOC2007/'          # TODO: change path to coco dataset
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    visualize = True
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

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
