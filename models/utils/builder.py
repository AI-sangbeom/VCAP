import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class VCAP(nn.Module):
    def __init__(self, backbone, prompter, head):
        super(VCAP, self).__init__()
        self.backbone = backbone
        self.prompter = prompter
        self.head = head
        
    def forward(self, x):
        x, feat = self.prompter(x)
        x = self.backbone(x, feat)
        # x = self.backbone(x)
        return self.head(x)
    
class VP(nn.Module):
    def __init__(self, backbone, prompter, head):
        super(VP, self).__init__()
        self.backbone = backbone
        self.prompter = prompter
        self.head = head

    def forward(self, x):        
        x = self.prompter(x)
        x = self.backbone(x)

        return self.head(x)
    

class Network(nn.Module):
    def __init__(self, backbone, head):
        super(Network, self).__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
    

class ddp(nn.Module):

    def __init__(self, model):
        super(ddp, self).__init__()
        self.module = model

    def forward(self, x):
        return self.module(x)


def model2device(env, model):
    model = model.cuda(env.device)
    if env.num_gpus == '1':
        return ddp(model)
    else:
        return DDP(model, device_ids=[env.device], find_unused_parameters=False)
    

def freeze(model):
    for name, params in model.named_parameters():
        if 'prompt' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

            
