from pprint import pprint


# Default Configs for training

class Config:

    # dataset params
    database = 'kitti'  # choose one from ['VOC', 'KITTI']
    voc_data_dir = '/data/ziruiw3/VOCdevkit/VOC2007/'
    kitti_data_dir = '/data/ziruiw3/KITTI2VOC/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    train_num_workers = 8
    test_num_workers = 8

    # optimizer params
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr = 1e-3
    lr_decay = 0.1

    # model params
    model = 'vgg16' 
    apply_fpn = False       # use feature pyramid network for region proposal
    deformable = False       # use deformable conv + deformable roi pooling
    deformable_groups = 1   # defromable groups (<= in_channels)

    # training params
    nms_thresh = 0.3        # iou threshold in nms
    score_thresh = 0.05     # score threshold in nms
    rpn_sigma = 3.          # rpn sigma for l1_smooth_loss
    roi_sigma = 1.          # roi sigma for l1_smooth_loss
    epoch = 14              # total training epoch
    epoch_decay = 9         # epoch to decay lr

    # testing params
    n_visual_imgs = 2   # number of images to visualize
    visualize = True
    plot_dir = './visuals'
    save_dir = './exp_deformable'

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
