from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--category', type=str, default='toothbrush')
parser.add_argument('--artifact', type=str, default='none')
parser.add_argument('--undeployed-uncorrected', type=str)
parser.add_argument('--undeployed-corrected', type=str)
parser.add_argument('--deployed-uncorrected', type=str)
parser.add_argument('--deployed-corrected', type=str)
parser.add_argument('--output-directory', type=str)
args = parser.parse_args()

import os
os.makedirs(args.output_directory, exist_ok=True)
os.makedirs(args.output_directory + '/undeployed/uncorrected', exist_ok=True)
os.makedirs(args.output_directory + '/undeployed/corrected', exist_ok=True)
os.makedirs(args.output_directory + '/deployed/uncorrected', exist_ok=True)
os.makedirs(args.output_directory + '/deployed/corrected', exist_ok=True)

from src.utils import data2img, immono, anomaly_boundary
from src.data import load_data_tensor, transform, target_transform
from src.artifacts import artifact_transform

import torch as tr
import matplotlib.pyplot as plt

transform_train = artifact_transform(args.artifact)

Xtrain, Xtest, ytest, Seg = load_data_tensor(args.category, transform=transform_train, target_transform=target_transform)

_, Xdeployed, _, _ = load_data_tensor(args.category, transform=transform, target_transform=target_transform)

Ruu = tr.load(args.undeployed_uncorrected)
Ruc = tr.load(args.undeployed_corrected)
Rdu = tr.load(args.deployed_uncorrected)
Rdc = tr.load(args.deployed_corrected)

# global_norm = (tr.cat([Ruu, Ruc, Rdu, Rdc])**4).mean(0).sum()**(1/4)
vmin, vmax = -.75, .75

for idx, (xu, xd, seg, ruu, ruc, rdu, rdc) in enumerate(zip(Xtest, Xdeployed, Seg, Ruu, Ruc, Rdu, Rdc)):
    seg = seg.detach().clone()
    norm = (tr.stack([ruu, ruc, rdu, rdc])**4).mean(0).sum()**(1/4)

    immono(anomaly_boundary(data2img(xu), seg).detach().numpy(), filename=args.output_directory + f'/{idx}_undeployed.png')
    immono(anomaly_boundary(data2img(xd), seg).detach().numpy(), filename=args.output_directory + f'/{idx}_deployed.png')

    immono(ruu / norm, filename=args.output_directory + f'/undeployed/uncorrected/{idx}.png', vmin=vmin, vmax=vmax)
    immono(ruc / norm, filename=args.output_directory + f'/undeployed/corrected/{idx}.png', vmin=vmin, vmax=vmax)
    immono(rdu / norm, filename=args.output_directory + f'/deployed/uncorrected/{idx}.png', vmin=vmin, vmax=vmax)
    immono(rdc / norm, filename=args.output_directory + f'/deployed/corrected/{idx}.png', vmin=vmin, vmax=vmax)
