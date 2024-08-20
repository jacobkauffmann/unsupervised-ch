from models import load_model
import torch
import numpy as np
from torchvision import transforms
from data import ImagenetSubset, FISH
from matplotlib.colors import ListedColormap
from torchvision.datasets import ImageFolder
from bilrp.data import load_github_dataset, load_nih_dataset
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from functools import partial

# Get the color map by name:
cm = plt.get_cmap('Greys')

my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
my_cmap[:, 0:3] *= 0.85
my_cmap = ListedColormap(my_cmap)

IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def edge_filter_transform(img, a=100, b=200):
    img = img.convert("L")
    img = np.array(img, dtype=np.uint8)
    # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
    edges = 0.0
    for k, w in zip([3, 4, 5, 6], [1, 2, 2, 1]):
        edges = edges + w * (cv.Canny(cv.blur(img, ksize=(k, k)), a, b))

    edges = edges / 255.0
    norm = plt.Normalize(vmin=edges.min(), vmax=edges.max())
    edges = cm(norm(edges))
    edges = (edges * 255).astype(np.uint8)
    return Image.fromarray(edges)


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    edge_filter_transform,
    transforms.ToTensor(),
])

transform_2 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def ToPIL(x):
    # pixels are in [-1024, 1024]
    x = x + 1024
    x = x * 255 / 2048
    # image is numpy
    x = Image.fromarray(np.uint8(x[0]))
    return x


transform_covid = transforms.Compose([
    ToPIL,
    transforms.Resize(224),
    transforms.CenterCrop(224),
    partial(edge_filter_transform, a=20, b=40),
    transforms.ToTensor(),
])

transform_covid_2 = transforms.Compose([
    ToPIL,
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

clip_norm = transforms.Normalize(*CLIP_NORM)
imagenet_norm = transforms.Normalize(*IMAGENET_NORM)


def get_fish_dataset(root='resources/data/subsets/fish/50-50/train', target=2):
    dataset_1 = ImagenetSubset(root=root, transform=transform, classes=FISH)
    dataset_2 = ImagenetSubset(root=root, transform=transform_2, classes=FISH)
    indices = np.argwhere(np.array(dataset_1.targets) == target).flatten()
    return torch.utils.data.Subset(dataset_1, indices), torch.utils.data.Subset(dataset_2, indices)


def get_covid_subset(github=True, label=1):
    if github:
        dataset_1 = load_github_dataset(transform=transform_covid)
        dataset_2 = load_github_dataset(transform=transform_covid_2)
    else:
        dataset_1 = load_nih_dataset(transform=transform_covid)
        dataset_2 = load_nih_dataset(transform=transform_covid_2)

    subset_indices = np.argwhere(np.array(dataset_1.labels) == label).flatten()
    print('subset_indices', subset_indices)
    return torch.utils.data.Subset(dataset_1, subset_indices), torch.utils.data.Subset(dataset_2, subset_indices)


def get_truck_dataset():
    dataset_1 = ImageFolder(root='resources/old/resources/lrp_images', transform=transform)
    dataset_2 = ImageFolder(root='resources/old/resources/lrp_images', transform=transform_2)
    return dataset_1, dataset_2


def get_covid_dataset():
    dataset = ImageFolder(root='resources/covid_data', transform=transform)
    return dataset


def load_models(dataset='fish', num_classes=16):
    if dataset.startswith('fish'):
        model_config = [('Sup-Fish', 'r50-sup'), ('CLIP', 'r50-clip-wo-attnpool'),
                        ('SimCLR', 'simclr-rn50'), ('BarlowTwins', 'r50-barlowtwins')]
    elif dataset == 'trucks':
        model_config = [('Sup-Truck', 'r50-sup'), ('CLIP', 'r50-clip-wo-attnpool'),
                        ('SimCLR', 'simclr-rn50'), ('BarlowTwins', 'r50-barlowtwins')]
    elif dataset.startswith('covid'):
        model_config = [('PubmedCLIP', 'pubmedclip')]
    else:
        raise ValueError()

    models = {}

    for name, identifier in model_config:
        if identifier == 'pubmedclip':
            model = load_model('r50-clip-wo-attnpool', model_paths={}, num_classes=num_classes)
            model = model.to('cpu')
            state_dict = torch.load('pubmedclip_RN50.pth', map_location=torch.device('cpu'))
            new_state_dict = {}
            for key, val in state_dict['state_dict'].items():
                new_state_dict[key.replace('visual.', 'encoder.')] = val
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model = load_model(identifier, model_paths={}, num_classes=num_classes)
            model = model.to('cpu')

        model.fc = torch.nn.Identity()
        model.eval()
        models[name] = model
    return models
