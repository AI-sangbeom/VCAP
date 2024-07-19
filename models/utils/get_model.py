import timm
import torch
from models.utils.builder import *
from models import *

def create_backbone(cfg):
    if cfg.model.name == 'vit_base_patch16_224':
        backbone = VisionTransformer(num_classes=21843)
    elif cfg.model.name == 'swin_base_patch4_window7_224':
        backbone = jx_swin_base_patch4_window7_224_in22k()
    else:
        backbone = timm.create_model(cfg.model.name,
                                    pretrained=cfg.model.pretrained, 
                                    drop_rate=cfg.train.dropout) 
    return backbone

def get_head(cfg, backbone):
    name = list(backbone._modules.keys())[-1]
    inf = backbone._modules[name].in_features
    head = nn.Linear(inf, cfg.data.num_classes)
    backbone._modules[name] = nn.Identity()
    return head

def get_FT(cfg):
    backbone = create_backbone(cfg)
    head = get_head(cfg, backbone)
    return Network(backbone, head)

def get_HT(cfg):
    backbone = create_backbone(cfg)
    freeze(backbone)
    head = get_head(cfg, backbone)
    return Network(backbone, head)

def get_VCAP(cfg):
    if cfg.model.name == 'vit_base_patch16_224':
        backbone = VisionTransformerwithFAP(num_classes=21843)
    elif cfg.model.name == 'swin_base_patch4_window7_224':
        backbone = jx_swin_base_patch4_window7_224_in22k_with_fap()   
    else:
        backbone = create_backbone(cfg)
    freeze(backbone)
    head = get_head(cfg, backbone)
    prompter = VisualChannelAdaptivePrompt(cfg)
    return VCAP(backbone, prompter, head)

def get_VP(cfg):
    backbone = create_backbone(cfg)
    freeze(backbone)
    head = get_head(cfg, backbone)
    prompter = PadPrompter(cfg)
    return VP(backbone, prompter, head)

def get_VPT(cfg):
    if cfg.model.name == 'vit_base_patch16_224':
        backbone = VisionTransformerwithPrompt(num_classes=21843)
    elif cfg.model.name == 'swin_base_patch4_window7_224':
        backbone = jx_swin_base_patch4_window7_224_in22k_with_fap()   
    else:
        backbone = create_backbone(cfg)
    freeze(backbone)
    head = get_head(cfg, backbone)
    return Network(backbone, head)

def get_base(cfg):
    if   cfg.method == 'FT'  : model = get_FT(cfg)
    elif cfg.method == 'HT'  : model = get_HT(cfg)
    elif cfg.method == 'VCAP': model = get_VCAP(cfg)
    elif cfg.method == 'VP'  : model = get_VP(cfg)
    elif cfg.method == 'VPT' : model = get_VPT(cfg)
    return model
   
def call_checkpoint(cfg, model):

    if cfg.model.name == 'vit_base_patch16_224':
        params = torch.load('checkpoints/vit_base_p16_224_in22k.pth')
        del(params['pre_logits.fc.bias'])
        del(params['pre_logits.fc.weight'])
        del(params['head.weight'])
        del(params['head.bias'])
        for i, j in model.backbone.named_parameters():
            if 'prompt' in i: params[i] = j
        model.backbone.load_state_dict(params)

    if cfg.model.name == 'swin_base_patch4_window7_224':
        params = torch.load('checkpoints/swin_base_patch4_window7_224_22k.pth')
        for i, j in model.backbone.named_parameters():
            if 'prompt' in i: params['model'][i] = j
        model.backbone.load_state_dict(params['model'])

    return model

def get_model(env):
    cfg = env.cfg
    if cfg.model.checkpoint != '':
        if env.p: print(' load checkpoint :', cfg.model.checkpoint)
        model = torch.load(cfg.model.checkpoint)
        freeze(model)
    else:
        model = get_base(cfg)
        model = call_checkpoint(cfg, model)

    return model2device(env, model)