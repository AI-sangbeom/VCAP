import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from TEST.dataset import dataset


def transform(data):
    img_size = 224
    if (data == 'mnist') or (data == 'fashion_mnist'):
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
    elif (data == 'inaturalist'):
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
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transform_train, transform_test

def dataLoader(data, batch, shuffle=True):
    transform_train, transform_test = transform(data)
    train_dataset, test_dataset = dataset(data, transform_train, transform_test)
    train_loader = DataLoader(train_dataset, 
                                batch_size=batch,
                                shuffle=shuffle)
    test_loader  = DataLoader(test_dataset, 
                                batch_size=batch,
                                shuffle=shuffle)
    return train_loader, test_loader


