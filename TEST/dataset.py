import torchvision
from lib.data.dataset import *
from lib.data.custom_datasets import d_set

def dataset(data, tr_transform, te_transform):
    
    dataset = data
    root = './../data'
    download = False

    if dataset=='cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=root,
                                           train=True,
                                           transform=tr_transform,
                                           download=download)
        test_dataset = torchvision.datasets.CIFAR10(root=root,
                                           train=False,
                                           transform=te_transform,
                                           download=download)
    elif dataset=='cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=root, 
                                              train=True, 
                                              transform=tr_transform, 
                                              download=download)

        test_dataset = torchvision.datasets.CIFAR100(root=root, 
                                             train=False, 
                                             transform=te_transform, 
                                             download=download)        
    elif dataset=='stanfordcars':
        
        train_dataset = torchvision.datasets.StanfordCars(root=root, 
                                                    split='train',
                                                    transform=tr_transform, 
                                                    download=download)

        test_dataset = torchvision.datasets.StanfordCars(root=root, 
                                                split='test',
                                                transform=te_transform, 
                                                download=download)
                
    elif dataset=='dtd':
        train_dataset = torchvision.datasets.DTD(root=root, 
                                                    split='train', 
                                                    transform=tr_transform, 
                                                    download=download)

        test_dataset = torchvision.datasets.DTD(root=root, 
                                                split='test', 
                                                transform=te_transform, 
                                                download=download)
    elif dataset=='flowers102':
        train_dataset = torchvision.datasets.Flowers102(root=root, 
                                                    split='train', 
                                                    transform=tr_transform, 
                                                    download=download)

        test_dataset = torchvision.datasets.Flowers102(root=root, 
                                                split='test', 
                                                transform=te_transform, 
                                                download=download)  
    
    elif dataset=='inaturalist':
        train_dataset = torchvision.datasets.INaturalist(root=root, 
                                                    version='2021_train', 
                                                    transform=tr_transform, 
                                                    target_type='full',
                                                    download=download)
 
        test_dataset = torchvision.datasets.INaturalist(root=root, 
                                                version='2021_valid', 
                                                transform=te_transform,
                                                target_type='full',
                                                download=download)
    elif dataset=='mnist':
        train_dataset = torchvision.datasets.MNIST(root=root, 
                                                    train=True, 
                                                    transform=tr_transform, 
                                                    download=download)
 
        test_dataset = torchvision.datasets.MNIST(root=root, 
                                                train=False, 
                                                transform=te_transform, 
                                                download=download)        
    elif dataset=='fashion_mnist':
        train_dataset = torchvision.datasets.FashionMNIST(root=root, 
                                                    train=True, 
                                                    transform=tr_transform, 
                                                    download=download)
 
        test_dataset = torchvision.datasets.FashionMNIST(root=root, 
                                                train=False, 
                                                transform=te_transform, 
                                                download=download)  
            
    elif dataset=='oxfordpet':
        train_dataset = torchvision.datasets.OxfordIIITPet(root=root, 
                                                    train=True, 
                                                    transform=tr_transform, 
                                                    target_types = 'category',
                                                    download=download)
 
        test_dataset = torchvision.datasets.OxfordIIITPet(root=root, 
                                                train=False, 
                                                transform=te_transform, 
                                                target_types = 'category',  
                                                download=download) 

    elif dataset=='food101':
        train_dataset = torchvision.datasets.Food101(root=root, 
                                                    split='train', 
                                                    transform=tr_transform, 
                                                    download=download)
 
        test_dataset = torchvision.datasets.Food101(root=root, 
                                                split='test', 
                                                transform=te_transform, 
                                                download=download) 
            
    elif dataset=='svhn':
        train_dataset = torchvision.datasets.SVHN(root=root, 
                                                    split='train', 
                                                    transform=tr_transform, 
                                                    download=download)
 
        test_dataset = torchvision.datasets.SVHN(root=root, 
                                                split='test', 
                                                transform=te_transform, 
                                                download=download)                 
            
    elif dataset=='gtsrb':
        train_dataset = torchvision.datasets.GTSRB(root=root, 
                                                    split='train', 
                                                    transform=tr_transform, 
                                                    download=download)
 
        test_dataset = torchvision.datasets.GTSRB(root=root, 
                                                split='test', 
                                                transform=te_transform, 
                                                download=download)     

    elif dataset=='nabirds':
        train_dataset = NABirds(root=root, train=True,
                                transform=tr_transform, 
                                download=download)
 
        test_dataset = NABirds(root=root, train=False,
                            transform=tr_transform, 
                            download=download)        
        
    elif dataset=='cub200':
        train_dataset = Cub2011(root=root, train=True,
                                transform=tr_transform, 
                                download=download)
 
        test_dataset = Cub2011(root=root, train=False,
                            transform=tr_transform, 
                            download=download)  

    elif dataset=='stanford_dogs':
        train_dataset = dogs(root=root, train=True,
                                transform=tr_transform, 
                                download=download)
 
        test_dataset = dogs(root=root, train=False,
                            transform=tr_transform, 
                            download=download)                                  
    elif dataset=='crc':
        train_dataset = d_set(data_dir='./../dataset/crc/train', 
                                load_bytes=False, 
                                transform=tr_transform)
        
            
        test_dataset = d_set(data_dir='./../dataset/crc/test', 
                                load_bytes=False, 
                                transform=te_transform)                                     
    if download: exit()
    else: return train_dataset, test_dataset