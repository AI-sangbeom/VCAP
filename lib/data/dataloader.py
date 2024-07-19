import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from lib.data.datasets import dataset
# from torch.utils.data import random_split

def transform(cfg):
    
    img_size = cfg.data.img_size
    if (cfg.data.dataset == 'mnist') or (cfg.data.dataset == 'fashion_mnist'):
        transform_train = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
        transform_test = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])       
    elif (cfg.data.dataset == 'inaturalist'):
        transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            # transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            # transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transform_train, transform_test

def dataLoader(env):
    cfg = env.cfg
    transform_train, transform_test = transform(cfg)
    train_dataset, test_dataset = dataset(env, transform_train, transform_test)
    shuffle = True
    if env.num_gpus=='1':
        train_loader = DataLoader(train_dataset, 
                                  batch_size=cfg.train.batch,
                                  shuffle=shuffle)
        test_loader  = DataLoader(test_dataset, 
                                  batch_size=cfg.train.batch,
                                  shuffle=shuffle)
    else:

        train_sampler = DistributedSampler(train_dataset,
                                        num_replicas=env.world_size,
                                        rank=env.rank,
                                        shuffle=shuffle,)                                
        test_sampler = DistributedSampler(test_dataset,
                                        num_replicas=env.world_size,
                                        rank=env.rank,
                                        shuffle=False,)

        train_loader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size = int(cfg.train.batch/env.world_size),
                                pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                sampler = test_sampler,
                                batch_size = int(cfg.train.batch/env.world_size),
                                pin_memory=True,)
    
    return train_loader, test_loader


