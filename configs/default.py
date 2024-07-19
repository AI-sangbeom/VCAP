from yacs.config import CfgNode as CN

C = CN()
C.method = 'VCAP'

C.data = CN()
C.data.root = './../data'
C.data.dataset = 'cifar10'
C.data.num_classes = 10
C.data.img_size = 224
C.data.download = False

C.model = CN()
C.model.name = 'resnet18'
C.model.pretrained = True
C.model.checkpoint = ''

C.prompt = CN()
C.prompt.hidden_dim = 20
C.prompt.bottle_neck = 10
C.prompt.size = 10

C.train = CN()
C.train.epoch = 50
C.train.batch = 64
C.train.lr = 1e-2
C.train.dropout = 0.
C.train.optimizer = 'adam'
C.train.momentum = 0.9
C.train.criterion = 'CE'

C.train.scheduler = CN()
C.train.scheduler.warmup_lr_init = 1e-3
C.train.scheduler.warmup_t = 5
C.train.scheduler.k_decay = 1.5
C.train.scheduler.lr_min = 1e-4


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

