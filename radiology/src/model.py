import clip
import torch as tr
import torch.nn as nn
import os
import torch
from src.chess_resnet import resnet50
from PIL import Image

# device = tr.device("mps" if tr.backends.mps.is_available() else "cpu")
device = tr.device("cpu")

def load_model(weights=None, **kwargs):
    if weights is None:
        weights = "RN50"
    model, preprocess = clip.load(weights, **kwargs, device=device)
    model.eval()
    return model, preprocess

### Chess Model
from torchvision.transforms import Compose, Resize, CenterCrop, Grayscale, ToTensor, Normalize

chess_preprocessing = Compose([
    Resize(size=224, interpolation=Image.BICUBIC, max_size=None, antialias=True),
    CenterCrop(size=(224, 224)),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize(mean=(0.485), std=(0.229))
])

def load_chess(weights=None):
    model = resnet50()
    if weights is None:
        pretrained_model = "src/weights/chess.pth.tar"
    else:
        pretrained_model = weights
    if pretrained_model is not None:
        if os.path.isfile(pretrained_model):
            print("=> loading checkpoint '{}'".format(pretrained_model))
            checkpoint = torch.load(pretrained_model, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(pretrained_model))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained_model))

        ##freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
    model.eval()
    return model
