import torch 
import torch.nn as nn
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import autocast
from tqdm import tqdm

class Base():

    def __init__(self):
        self.model = None
        self.trainLoader = None
        self.testLoader = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.vp_optimizer = None
        self.scaler = None
        self.env = None

    def model_train(self):
        self.model.train()
        self.m_train()
        
    def model_valid(self, ):
        self.model.eval()
        self.m_valid()

    def m_train(self):
        env = self.env
        correct = 0
        total = 0
        rloss = 0
        accuracy = 0

        pbar = tqdm(self.trainLoader,
                    total=len(self.trainLoader),
                    ncols=80,
                    leave=True) if env.p else self.trainLoader
                
        for i, data in enumerate(pbar):
            
            if env.p: pbar.set_description(f'Epoch {self.epoch+1}')
            images, labels = data
            images, labels = images.to(env.device), labels.to(env.device)
            
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred==labels).sum().item()
            rloss += loss.item()
            running_loss = rloss/(i+1) 
            accuracy = 100*(correct/total)
            if env.p: pbar.set_postfix(acc=f"{accuracy:.1f}", loss=f"{running_loss:.2f}")

        self.acc = accuracy
        self.loss = running_loss
        self.scheduler.step(self.epoch)
        
    
    def m_valid(self):
        env = self.env
        self.model.eval()
        correct = 0
        total = 0
        rloss =0 
        accuracy =0 
        pbar = tqdm(self.testLoader, 
                    total=len(self.testLoader), 
                    ncols = 80, 
                    leave=True) if env.p else self.testLoader

        with torch.no_grad():
            
            for i, data in enumerate(pbar):
                if env.p: pbar.set_description(f'Valid {self.epoch+1}')
                images, labels = data
                images, labels = images.to(env.device), labels.to(env.device)
                outputs = self.model(images)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                loss = self.criterion(outputs, labels)
                rloss += loss.item()
                running_loss = rloss/(i+1) 
                correct += (pred==labels).sum().item()
                accuracy = (correct/total)*100
                if env.p: pbar.set_postfix(acc=f"{accuracy:.1f}", loss=f"{running_loss:.2f}")

        self.val_acc = accuracy
        self.val_loss = running_loss
        if env.p: print('')

    def init_numerical_value(self):
        self.epoch = 0
        self.best = 0
        self.acc = 0
        self.val_acc = 0
        self.loss = 0
        self.val_loss = 0    

    def save_best(self):
        if self.val_acc > self.best:
            self.best = self.val_acc
            torch.save(self.model.module, self.env.checkpoint)

def optimizer(cfg, model):
    if cfg.train.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr = cfg.train.lr)
    elif cfg.train.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = cfg.train.lr, momentum=cfg.train.momentum)
    return optimizer

def criterion(cfg):
    if cfg.train.criterion == 'CE':
        criterion = nn.CrossEntropyLoss()
    return criterion

def scheduler(cfg, optimizer):
    t_cfg = cfg.train.scheduler
    scheduler = CosineLRScheduler(optimizer,
                                    t_initial=cfg.train.epoch,
                                    cycle_decay=0.5,
                                    lr_min=t_cfg.lr_min,
                                    k_decay=t_cfg.k_decay,
                                    t_in_epochs=True, 
                                    warmup_t=t_cfg.warmup_t,
                                    warmup_lr_init=t_cfg.warmup_lr_init,
                                    cycle_limit=1)
    return scheduler