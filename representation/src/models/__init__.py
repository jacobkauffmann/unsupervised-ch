import torch
from collections import OrderedDict
from torchvision.models import resnet50, resnet18
from models.clip import load_clip_rn50, load_clip_rn50_wo_attnpool, load_clip_rn50_detach
from models.utils import load_vissl_r50
from models.vit import load_vit_model
from models.vissl import VisslLoader


def load_r50_checkpoint(model_path, delete_prefix='model.',
                        state_dict_key='state_dict', classes=10, clip=False):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in model[state_dict_key].items():
        new_key = key.replace(delete_prefix, '')
        new_state_dict[new_key] = value
    if clip:
        resnet = load_clip_rn50(classes)
    else:
        resnet = resnet50(zero_init_residual=True)
        resnet.fc = torch.nn.Linear(2048, classes)
    msg = resnet.load_state_dict(new_state_dict, strict=False)
    print(msg)
    return resnet


def load_model(name, model_paths, num_classes=10):
    if name in VisslLoader.MODELS.keys():
        loader = VisslLoader(name)
        backbone = loader.load_model_from_source()
    elif name == 'r50-barlowtwins':
        backbone = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    elif name == 'r50-swav':
        backbone = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    elif name == 'r50-vicreg':
        backbone = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    elif name == 'r50-clip':
        backbone = load_clip_rn50(num_classes=num_classes)
    elif name == 'r50-clip-wo-attnpool':
        backbone = load_clip_rn50_wo_attnpool(num_classes=num_classes)
    elif name == 'r50-clip-detach':
        backbone = load_clip_rn50_detach(num_classes=num_classes)
    elif name == 'vit-b16-mae':
        chkpt_dir = 'resources/mae_pretrain_vit_base.pth'
        backbone = load_vit_model(chkpt_dir, 'vit_base_patch16', state_dict_key='model')
    elif name == 'r50-sup':
        backbone = resnet50(pretrained=True)
    elif name == 'r50-scratch':
        backbone = resnet50(pretrained=False)
    elif name == 'r18-sup':
        backbone = resnet18(pretrained=True)
    elif name in model_paths:
        backbone = load_r50_checkpoint(model_paths[name], classes=num_classes, clip=name.startswith('r50-clip'))
    else:
        raise ValueError()
    return backbone
