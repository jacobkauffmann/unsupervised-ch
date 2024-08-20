import random

from bilrp.utils import (load_models, get_fish_dataset, imagenet_norm, clip_norm, get_truck_dataset,
                         get_covid_dataset, get_covid_subset)
from bilrp.bilrp import compute_branch, pool
from zennit.torchvision import ResNetCanonizer
from utils.lrp import module_map_resnet
from zennit.core import Composite
import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--model')
parser.add_argument('--output', default='bilrp/relevances')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

device = args.device

random.seed(42)
if args.dataset == 'fish-tench':
    _, dataset = get_fish_dataset(target=2)
    indices = [1, 3, 4, 5, 8, 9, 12, 13, 16, 17, 18, 19]
elif args.dataset == 'fish-coho':
    _, dataset = get_fish_dataset(target=13)
    indices = [2, 12, 14, 18, 19, 21, 25, 27, 29, 31, 32, 45, 47, 51]
elif args.dataset == 'trucks':
    _, dataset = get_truck_dataset()
    indices = [4, 5, 6, 7, 8]
elif args.dataset == 'covid':
    _, dataset = get_covid_dataset()
    indices = list(range(len(dataset)))
elif args.dataset == 'covid-github-1':
    _, dataset = get_covid_subset(github=True, label=1)
    indices = random.sample(list(range(len(dataset))), k=30)
elif args.dataset == 'covid-github-0':
    _, dataset = get_covid_subset(github=True, label=0)
    indices = random.sample(list(range(len(dataset))), k=30)
elif args.dataset == 'covid-nih-1':
    _, dataset = get_covid_subset(github=False, label=1)
    indices = random.sample(list(range(len(dataset))), k=30)
elif args.dataset == 'covid-nih-0':
    _, dataset = get_covid_subset(github=False, label=0)
    indices = random.sample(list(range(len(dataset))), k=30)
else:
    raise ValueError()

print('indices', indices)

canonizers = [ResNetCanonizer()]
composite = Composite(module_map=module_map_resnet, canonizers=canonizers)

models = load_models(dataset=args.dataset)
model = models[args.model].to(device)

output_path = os.path.join(args.output, args.dataset, args.model)
os.makedirs(output_path, exist_ok=True)

for image_idx in tqdm(indices):
    sample = dataset[image_idx]
    if isinstance(sample, list) or isinstance(sample, tuple):
        sample = sample[0]
    if isinstance(sample, dict):
        sample = sample['img']

    if sample.shape[0] == 1:
        sample = sample.repeat(3, 1, 1)

    if args.model in ['CLIP', 'PubmedCLIP']:
        x = clip_norm(sample)
    else:
        x = imagenet_norm(sample)
    x = x.unsqueeze(0)
    x.requires_grad = True
    x = x.to(device)
    R, _ = compute_branch(x, model, composite)
    np.save(os.path.join(output_path, f'{image_idx}'), np.stack(R))
