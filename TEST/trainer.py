import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from torch.cuda.amp import autocast
from timm.scheduler.cosine_lr import CosineLRScheduler

def train(epoch, model, trainLoader, optimizer, criterion, scheduler, device):
    
    model.train()
    correct = 0
    total = 0
    running_loss = 0
    accuracy = 0
    pbar = tqdm(trainLoader,
                total=len(trainLoader),
                ncols=80,
                leave=True)
    
    for data in pbar:

        pbar.set_description(f'Epoch {epoch+1}')
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
    
        loss.backward()
        optimizer.step()
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred==labels).sum().item()
        running_loss = loss.item()
        accuracy = 100*(correct/total)
        pbar.set_postfix(acc=accuracy, loss=running_loss)

    scheduler.step(epoch)

def valid(model, testLoader, criterion, device):
    
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    accuracy = 0
    pbar = tqdm(testLoader,
                total=len(testLoader),
                ncols=80,
                leave=True)
    
    with torch.no_grad():
        for data in pbar:
            pbar.set_description(f'Valid')
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            loss = criterion(outputs, labels)
            running_loss = loss.item()
            correct += (pred==labels).sum().item()
            accuracy = (correct/total)*100
            pbar.set_postfix(acc=accuracy, loss=running_loss)

def get_optimizer(params, optim_type, LR):
    if optim_type == 'adam':
        optimizer = optim.Adam(params, lr=LR)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(params, lr=LR, momentum=0.9)
    return optimizer

def get_criterion():
    return nn.CrossEntropyLoss()

def get_scheduler(optimizer, t_initial, cycle_decay=0.5, lr_min=1e-3, k_decay=1., warmup_t=5, warmup_lr_init=1e-2):
    scheduler = CosineLRScheduler(optimizer,
                                    t_initial=t_initial,
                                    cycle_decay=cycle_decay,
                                    lr_min=lr_min,
                                    k_decay=k_decay,
                                    t_in_epochs=True, 
                                    warmup_t=warmup_t,
                                    warmup_lr_init=warmup_lr_init,
                                    cycle_limit=1)
    return scheduler