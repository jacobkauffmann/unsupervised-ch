from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--category', type=str, default='toothbrush')
parser.add_argument('--output', type=str)
args = parser.parse_args()

from src.models import MLP
from src.data import load_data_tensor, CATEGORIES, transform, target_transform
from src.utils import split_for_supervised

Xtrain, Xtest, ytest, Seg = load_data_tensor(args.category, transform=transform, target_transform=target_transform)

Xtrain, Xtest, ytrain, ytest = split_for_supervised(Xtrain, Xtest, ytest, test_size=0.8, random_state=42)

model = MLP()
model.fit(Xtrain, ytrain)

import matplotlib.pyplot as plt
import numpy as np

# 10x10 grid of weights as grayscale images
# weights = model.linear1.weight.view(10, 10, 3, 64, 64).detach().cpu().numpy()
# it's now a conv layer with 100 kernels of size 50x50
weights = model.conv1.weight.view(10, 10, 3, 50, 50).detach().cpu().numpy()
weights = weights.sum(2)

fig, axs = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(weights[i, j], cmap='gray', vmin=weights.min(), vmax=weights.max())
        axs[i, j].axis('off')
plt.tight_layout()
plt.savefig(args.output)
