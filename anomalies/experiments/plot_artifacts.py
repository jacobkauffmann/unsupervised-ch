from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--artifact', type=str, default='none')
parser.add_argument('--category', type=str, default='toothbrush')
parser.add_argument('--output-directory', type=str, default='results')
parser.add_argument('--n-plot-artifacts', type=int, default=5)
args = parser.parse_args()

from src.artifacts import artifacts, artifact_transform
from src.data import load_data_tensor, CATEGORIES, transform
from src.utils import immono, data2img

import sklearn
import matplotlib.pyplot as plt
import torch as tr
import numpy as np
import os

if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)

_, Xtest_artifact, _, _ = load_data_tensor(args.category, transform=artifact_transform(args.artifact), target_transform=transform)
for idx, x in enumerate(Xtest_artifact[:args.n_plot_artifacts]):
    x = x.detach().cpu().numpy()
    immono(data2img(x).numpy(), filename=f'{args.output_directory}/{idx}.png')
