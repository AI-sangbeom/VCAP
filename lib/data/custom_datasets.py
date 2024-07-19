import os
import torch
import re
import natsort
from torch.utils.data import Dataset
from PIL import Image

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def find_image_target(folder, types=IMG_EXTENSIONS, leaf_name_only=True, sort=True):
    labels = []
    filenames = [] 
    for root, _, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)

    unique_labels = set(labels)
    sorted_labels = list(sorted(unique_labels, key=natural_key))
    class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx

class d_set(Dataset):
    def __init__(self, data_dir, load_bytes=False, transform=None):

        images, class_to_idx = find_image_target(data_dir)

        if len(images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + data_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.data_dir = data_dir
        self.samples = images
        self.imgs = self.samples
        self.class2idx = class_to_idx
        self.transform = transform
        self.load_bytes = load_bytes

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]
            
    def _extract_tar_info(tarfile, class_to_idx=None, sort=True):
        files = []
        labels = []
        for ti in tarfile.getmembers():
            if not ti.isfile():
                continue
            dirname, basename = os.path.split(ti.path)
            label = os.path.basename(dirname)
            ext = os.path.splitext(basename)[1]
            if ext.lower() in IMG_EXTENSIONS:
                files.append(ti)
                labels.append(label)
        if class_to_idx is None:
            unique_labels = set(labels)
            sorted_labels = list(sorted(unique_labels, key=natural_key))
            class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
        tarinfo_and_targets = zip(files, [class_to_idx[l] for l in labels])
        if sort:
            tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
        return tarinfo_and_targets, class_to_idx

# Dataset


class LT_Dataset(Dataset):
    
    def __init__(self, root, dataset, type, transform=None):
        root = './data/%s/%s_open'%(dataset, dataset)
        txt = './data/%s/%s_open.txt'%(dataset, dataset)
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path
    
class OCR(Dataset):

    def __init__(self, path, train=True, transform=None):
        super(OCR, self).__init__()
        if train:
            self.path = os.path.join(path, 'OCR', 'Training', 'images')
        else:
            self.path = os.path.join(path, 'OCR', 'Validation', 'images')
        self.target_labeling(self.path)
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

    def target_labeling(self, path):
        labels = natsort.natsorted(os.listdir(path))
        self.data = {}
        self.class2idx = {}
        self.idx2class = {}
        i = 0
        for label, target in enumerate(labels):
            class_img = os.path.join(path, target)
            self.class2idx[target] = label
            self.idx2class[label] = target
            for img in os.listdir(class_img):
                self.data[i] = [os.path.join(class_img, img), label]
                i+=1

    def classtoidx(self, target):
        return self.class2idx[target]
    
    def idxtoclass(self, label):
        return self.idx2class[label]
    