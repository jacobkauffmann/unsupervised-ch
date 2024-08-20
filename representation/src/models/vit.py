from models import models_vit
import torch


def load_vit_model(chkpt_dir, arch='vit_base_patch16', state_dict_key=None):
    model = getattr(models_vit, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    state_dict = checkpoint[state_dict_key] if state_dict_key is not None else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    return model
