import torch
import torch.nn as nn

class PadPrompter(nn.Module):
    def __init__(self, cfg):
        super(PadPrompter, self).__init__()
        pad_size = int(cfg.prompt.size)
        self.pad_size = pad_size
        image_size = cfg.data.img_size
        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size-pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size-pad_size*2, pad_size]))
        self.norm = nn.BatchNorm2d(3)

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0)*[prompt])

        return self.norm(x + prompt)
    
    