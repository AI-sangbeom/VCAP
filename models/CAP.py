import torch
import torch.nn as nn
import timm

class ChannelSelector(nn.Module):
    def __init__(self, hidden_dim, bottle_neck):
        super(ChannelSelector, self).__init__()
        self.FE = timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes=bottle_neck)
        head = list(self.FE._modules.keys())[-1]
        for name, params in self.FE.named_parameters():
            if head in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

        self.FE_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bottle_neck, hidden_dim)
        )

        self.select_rgb = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
    def forward(self, x):
        feat = self.FE_proj(self.FE(x))
        B, C = feat.size()
        prompt = self.select_rgb(feat).reshape(B, 3, 1, 1).expand([B, 3, 224, 224])
        return prompt, feat.reshape(B,1,C)
    

class VisualChannelAdaptivePrompt(nn.Module):
    def __init__(self, cfg):
        super(VisualChannelAdaptivePrompt, self).__init__()
        # hidden_dim = cfg.prompt.hidden_dim
        # bottle_neck = cfg.prompt.bottle_neck
        hidden_dim = 40
        bottle_neck = 20
        self.CV1 = nn.Sequential(     
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.ReLU(),
        )
        self.CV2 = nn.Sequential(     
            nn.Tanh(),
            nn.Conv2d(3, 3, 3, 1, 1),
        )
        self.CS1 = ChannelSelector(hidden_dim, bottle_neck)
        self.CS2 = ChannelSelector(hidden_dim, bottle_neck)
        self.gf_proj1 = nn.Sequential(
            nn.Conv1d(1, 49, 1, 1),            
            nn.Linear(hidden_dim, bottle_neck),
        )
        self.gf_proj2 = nn.Sequential(
            nn.Conv1d(1, 49, 1, 1),            
            nn.Linear(hidden_dim, bottle_neck),
        )
    def forward(self, x):
        prompt, global_feature = self.ConvSE(x)
        return prompt + x, global_feature
        
    def ConvSE(self, x):
        prompt, gf1 = self.CS1(x)
        x = self.CV1(x + prompt)
        prompt, gf2 = self.CS2(x)
        x = self.CV2(x + prompt)
        return x, self.gf_proj1(gf1) + self.gf_proj2(gf2)