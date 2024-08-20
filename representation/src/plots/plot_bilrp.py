from bilrp.utils import (get_fish_dataset, get_truck_dataset,
                         get_covid_dataset, get_covid_subset)
from bilrp.bilrp import plot_bilrp
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-1')
parser.add_argument('--dataset-2')
parser.add_argument('--output', default='bilrp/images')
parser.add_argument('--random', action='store_true')
parser.add_argument('--normalization-factor', default='individual')
args = parser.parse_args()

folder = args.output

datasets = []
datasets_2 = []
dataset_indices = []
for ds in [args.dataset_1, args.dataset_2]:
    if ds == 'fish-tench':
        dataset, dataset2 = get_fish_dataset()
        indices = [1, 3, 4, 5, 8, 9, 12, 13, 16, 17, 18, 19]
    elif ds == 'fish-coho':
        dataset, dataset2 = get_fish_dataset(target=13)
        indices = [2, 12, 14, 18, 19, 21, 25, 27, 29, 31, 32, 45, 47, 51]
    elif ds == 'trucks':
        dataset, dataset2 = get_truck_dataset()
        indices = [4, 5, 6, 7, 8]
    elif ds == 'covid':
        dataset = get_covid_dataset()
        indices = list(range(len(dataset)))
    elif ds == 'covid-github-1':
        dataset, dataset2 = get_covid_subset(github=True, label=1)
        indices = random.sample(list(range(len(dataset))), k=30)
    elif ds == 'covid-github-0':
        dataset, dataset2 = get_covid_subset(github=True, label=0)
        indices = random.sample(list(range(len(dataset))), k=30)
    elif ds == 'covid-nih-1':
        dataset, dataset2 = get_covid_subset(github=False, label=1)
        indices = random.sample(list(range(len(dataset))), k=30)
    elif ds == 'covid-nih-0':
        dataset, dataset2 = get_covid_subset(github=False, label=0)
        indices = random.sample(list(range(len(dataset))), k=30)
    else:
        raise ValueError()
    datasets.append(dataset)
    datasets_2.append(dataset2)
    dataset_indices.append(indices)

relevance_dir = 'bilrp/relevances'
model = list(os.listdir(os.path.join(relevance_dir, args.dataset_1)))[0]
indices_1 = list(os.listdir(os.path.join(relevance_dir, args.dataset_1, model)))
indices_2 = list(os.listdir(os.path.join(relevance_dir, args.dataset_2, model)))

indices_1 = [int(idx.split('.')[0]) for idx in indices_1]
indices_2 = [int(idx.split('.')[0]) for idx in indices_2]

print(indices_1)
print(indices_2)

if args.random:
    idx_1 = random.choice(indices_1)
    idx_2 = random.choice(indices_2)
    indices = [(idx_1, idx_2)]
else:
    if args.dataset_1 == 'fish':
        indices = [(13, 3), (4, 16)]
    else:
        indices = [(5, 7), (6, 4), (5, 6)]

for idx_1, idx_2 in indices:

    def transform_sample(sample):
        if isinstance(sample, list) or isinstance(sample, tuple):
            sample = sample[0]
        if isinstance(sample, dict):
            sample = sample['img']
        return sample


    for model in os.listdir(os.path.join(relevance_dir, args.dataset_1)):
        R1 = np.load(os.path.join(relevance_dir, args.dataset_1, model, f'{idx_1}.npy'))
        R2 = np.load(os.path.join(relevance_dir, args.dataset_2, model, f'{idx_2}.npy'))

        x1 = transform_sample(datasets[0][idx_1])
        x2 = transform_sample(datasets[1][idx_2])
        x1 = torch.permute(x1, (1, 2, 0))
        x2 = torch.permute(x2, (1, 2, 0))

        out_name = args.dataset_1 if args.dataset_1 == args.dataset_2 else f"{args.dataset_1}_{args.dataset_2}"

        os.makedirs(os.path.join(folder, out_name, f'{idx_1}-{idx_2}'), exist_ok=True)
        fname = os.path.join(folder, out_name, f'{idx_1}-{idx_2}', f'{model}.png')
        plot_bilrp(x1, x2, R1, R2, fname=fname, normalization_factor=args.normalization_factor)
        plt.clf()

        x1 = transform_sample(datasets_2[0][idx_1])
        x2 = transform_sample(datasets_2[1][idx_2])
        x1 = torch.permute(x1, (1, 2, 0))
        x2 = torch.permute(x2, (1, 2, 0))
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        if args.dataset_1.startswith('covid'):
            ax[0].imshow(x1, cmap='Greys')
            ax[1].imshow(x2, cmap='Greys')
        else:
            ax[0].imshow(x1)
            ax[1].imshow(x2)

        h, w, channels = x1.shape if len(x1.shape) == 3 else list(x1.shape) + [1]
        wgap, hpad = int(0.05 * w), int(0.6 * w)
        plt.subplots_adjust(hspace=wgap)

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(folder, out_name, f'{idx_1}-{idx_2}', f'images.png'), transparent=True)
