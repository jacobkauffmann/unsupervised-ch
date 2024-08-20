from .clip import load
import torch
from models.clip.model import AttentionPool2dDetach


class ModelWrapper(torch.nn.Module):

    def __init__(self, encoder, num_classes=10, ga_pooling=False):
        super().__init__()
        self.encoder = encoder
        self.fc = torch.nn.Linear(1024, num_classes)
        self.ga_pooling = ga_pooling

    def forward(self, x):
        rep = self.encoder(x)
        if self.ga_pooling:
            rep = torch.mean(rep.view(rep.size(0), rep.size(1), -1), dim=2)
        return self.fc(rep)


def load_clip_rn50(num_classes=10):
    model, transform = load(name='RN50', device='cpu')
    model = model.visual
    return ModelWrapper(encoder=model, num_classes=num_classes)


def load_clip_rn50_wo_attnpool(num_classes=10):
    model, transform = load(name='RN50', device='cpu')
    model = model.visual
    model.attnpool = torch.nn.Identity()
    return ModelWrapper(encoder=model, num_classes=num_classes, ga_pooling=True)


def load_clip_rn50_detach(num_classes=10):
    model, transform = load(name='RN50', device='cpu')
    model = model.visual
    model.attnpool = AttentionPool2dDetach(positional_embedding=model.attnpool.positional_embedding,
                                           c_proj=model.attnpool.c_proj,
                                           v_proj=model.attnpool.v_proj,
                                           k_proj=model.attnpool.k_proj,
                                           q_proj=model.attnpool.q_proj,
                                           num_heads=model.attnpool.num_heads)
    return ModelWrapper(encoder=model, num_classes=num_classes, ga_pooling=False)
