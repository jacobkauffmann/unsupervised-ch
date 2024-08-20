import sys

sys.path.append('.')

import os
import random
from models import load_model
import torch
import numpy as np
from torchvision import transforms
from data import ImagenetSubset, FISH
from zennit.torchvision import ResNetCanonizer
from zennit.attribution import Gradient
from utils.lrp import module_map_resnet
from zennit.core import Composite
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['trucks', 'fish'])
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()

my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
my_cmap[:, 0:3] *= 0.85
my_cmap = ListedColormap(my_cmap)

IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

clip_norm = transforms.Normalize(*CLIP_NORM)
imagenet_norm = transforms.Normalize(*IMAGENET_NORM)

model_config = [('CLIP', 'r50-clip-wo-attnpool'),
                ('SimCLR', 'simclr-rn50'),
                ('BT', 'r50-barlowtwins'),
                ('Supervised', 'r50-sup')]

if args.dataset == 'fish':
    dataset = ImagenetSubset(root='resources/data/subsets/fish/50-50/train', transform=transform, classes=FISH)
    target_cls = 13  # coho
    num_classes = 16
    indices = np.argwhere(np.array(dataset.targets) == target_cls).flatten()
    dataset = torch.utils.data.Subset(dataset, indices)
elif args.dataset == 'trucks':
    dataset = ImageFolder(root='resources/old/resources/lrp_images', transform=transform)
    target_cls = 1
    num_classes = 8
else:
    raise ValueError()

models = {}
for name, identifier in model_config:
    model = load_model(identifier, model_paths={}, num_classes=num_classes)
    model = model.to('cpu')

    if identifier == 'r50-clip':
        linear = torch.nn.Linear(1024, num_classes, bias=False)
    else:
        linear = torch.nn.Linear(2048, num_classes, bias=False)

    coefs = np.load(f'resources/linear_probe/{args.dataset}/{identifier}_probe_weights.npz')
    linear.weight.data = torch.tensor(coefs[identifier], dtype=torch.float32)
    model.fc = linear

    model.eval()
    models[name] = model


def attr_output_fn(output, target):
    # output times one-hot encoding of the target labels of size (len(target), 1000)
    return output * torch.eye(num_classes)[target]


indices = list(range(len(dataset)))
random.shuffle(indices)
indices = indices[:min(20, len(indices))]
for k in indices:
    sample, target = dataset[k]
    maps = {}
    outputs = {}

    for model_name, model in models.items():
        if model_name.startswith('CLIP'):
            input = clip_norm(sample)
        else:
            input = imagenet_norm(sample)

        input = input.unsqueeze(0)
        input.requires_grad = True
        model.zero_grad()
        input.grad = None
        canonizers = [ResNetCanonizer()]
        composite = Composite(module_map=module_map_resnet, canonizers=canonizers)
        indices = []

        with Gradient(model, composite) as attributor:
            output_relevance = partial(attr_output_fn, target=target_cls)
            output, relevance = attributor(input, output_relevance)
            relevance = relevance.squeeze(0).sum(axis=0).numpy()
            maps[model_name] = relevance
            outputs[model_name] = F.softmax(output, dim=-1).detach().numpy()[0]

    fig, ax = plt.subplots(1, 1 + len(models))
    fig.set_size_inches((30, 15))
    fig.tight_layout()
    ax[0].imshow(torch.permute(sample.detach(), (1, 2, 0)))

    for idx, (name, attribution) in enumerate(maps.items()):
        b = 8.0 * ((np.abs(attribution) ** 3.0).mean() ** (1.0 / 3))
        ax[idx + 1].imshow(attribution, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')

    for a in ax:
        a.axis('off')

    os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, args.dataset, f'{k}.png'), transparent=True, bbox_inches='tight')
