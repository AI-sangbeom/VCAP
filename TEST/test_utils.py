import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import natsort
import pandas as pd
from tqdm import tqdm
from TEST.dataloader import dataLoader

def get_data(data, BS, cnt=0, mode='train'):
    trainloader, testloader = dataLoader(data, BS)
    k = 0
    if mode=='train':
        for img, label in trainloader:
            img = img.cuda()
            label = label.cuda()
            if k == cnt: break
    else:
        for img, label in testloader:
            img = img.cuda()
            label = label.cuda()
            if k == cnt: break
    return img, label


def freeze(model):
    for _, params in model.named_parameters():
        params.requires_grad = False

def get_head(backbone, num_classes):
    name = list(backbone._modules.keys())[-1]
    inf = backbone._modules[name].in_features
    head = nn.Linear(inf, num_classes)
    del(backbone._modules[name])
    return head

def get_prompt_img(dataset, BS, base_path='output', model='vit_base_patch16_224'):
    trainLoader, _ = dataLoader(dataset, BS, False)
    for img, _ in trainLoader:
        img = img.cuda()
        break
    x = img
    base = os.path.join(base_path, model, dataset)
    pth = sorted(os.listdir(base))[-2]
    p = os.path.join(base, pth)
    model = torch.load(p).cuda()
    feat, _ = model.prompter.ConvSE(img)
    out, _ = model.prompter(img)
    
    images = [img, feat, out]
    return images

def show_images(images, cnt=1, show_bs=10):
    
    s = len(images)
    for k in range(show_bs*(cnt-1), show_bs*(cnt)):
        plt.figure(figsize = (20, 5))
        
        for i in range(s):
            
            plt.subplot(1, s+1, i+1)
            plt.gca().axes.xaxis.set_visible(False)
            plt.gca().axes.yaxis.set_visible(False)    
            plt.imshow(images[i][k].permute(1, 2, 0).detach().cpu().numpy())
            
        plt.show()


def prompted_data_dist(data, BATCH, base_path='output'):
    trainLoader, _ = dataLoader(data, BATCH, False)
    data_len = trainLoader.dataset.__len__()
    base = os.path.join(base_path, 'embedding', data, 'vit_base_patch16_224/pad_size_10')
    pth = sorted(os.listdir(base))[-4]
    p = os.path.join(base, pth)
    model = torch.load(p).cuda()
    pbar = tqdm(trainLoader, total=len(trainLoader),
                desc=data, ncols=80, leave=True) 
    for n, (i, _) in enumerate(pbar):
        i = i.cuda()
        i = model.prompter(i)
        if n == 0: distribution = i.sum(0).detach().cpu()
        else: distribution += i.sum(0).detach().cpu()
    return distribution/data_len

def original_data_dist(data, BATCH):
    trainLoader, _ = dataLoader(data, BATCH, False)
    data_len = trainLoader.dataset.__len__()
    pbar = tqdm(trainLoader, total=len(trainLoader),
                desc=data, ncols=80, leave=True) 
    for n, (i, _) in enumerate(pbar):
        if n == 0: distribution = i.sum(0)
        else: distribution += i.sum(0)
    return distribution/data_len



def acc_log(path):
    file = path
    acc = []
    loss = []
    five = 0 
    ten = 0
    fiteen = 0
    with open(file, 'r') as f:
        texts = f.readlines()
        
        for t in texts:
            if 'Epoch' in t:
                ls = float(t[t.rfind('=')+1:])
                val = t[t.find('valid')+1:]
                val = float(val[val.find('=')+1:val.find(',')])
                acc.append(val)
                loss.append(ls)
                
                if '[Epoch 5]' in t:
                    five = val
                if '[Epoch 10]' in t:
                    ten = val
                if '[Epoch 50]' in t:
                    fiteen = val
        best = max(acc)
        
        return five, ten, fiteen, best


def get_param(datasets, base_path='output'):
    tt = 0
    for dataset in datasets:
        base = os.path.join(base_path, '/embedding', dataset, 'vit_base_patch16_224/pad_size_10')
        pth = sorted(os.listdir(base))[-1]
        file = os.path.join(base, pth)
        
        with open(file, 'r') as f:
            texts = f.readlines()

            for t in texts:
                if 'Total' in t:
                    idx = t.rfind(' ')
                    params = int(t[idx:].replace(',','').replace('\n',''))
                    tt += params
                    print(params)
                    break
    print(tt)
                    
            
def output_log(datas, base_path, model_name, cnt=-1, save=True):
    totals = [[], [], [], []]
    for data in datas:
        file = os.path.join(base_path, model_name, data)
        checkpoint = natsort.natsorted(os.listdir(file))[cnt]
        path = os.path.join(file, checkpoint)
        five, ten, fiteen, best = acc_log(path)
        totals[0].append(five)
        totals[1].append(ten)
        totals[2].append(fiteen)
        totals[3].append(best)
    
    totals = torch.Tensor(totals).T
    mean = totals.mean(0).unsqueeze(0)
    totals = torch.cat([totals, mean], dim=0)
    datas.append('average')
    datasets = dict(zip(datas, totals))
    df = pd.DataFrame(datasets)
    if save:
        df.to_csv('output.csv')
    return df