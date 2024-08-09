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

# for idx, (xu, xd, seg, ruu, ruc, rdu, rdc) in enumerate(zip(Xtest, Xdeployed, Seg, Ruu, Ruc, Rdu, Rdc)):
for idx, (xu, xd, seg, ruu_low, ruc_low, rdu_low, rdc_low, ruu_medium, ruc_medium, rdu_medium, rdc_medium, ruu_high, ruc_high, rdu_high, rdc_high) in enumerate(zip(Xtest, Xdeployed, Seg, Ruu['low'], Ruc['low'], Rdu['low'], Rdc['low'], Ruu['medium'], Ruc['medium'], Rdu['medium'], Rdc['medium'], Ruu['high'], Ruc['high'], Rdu['high'], Rdc['high'])):
    seg = seg.detach().clone()
    norm = (tr.stack([ruu_low, ruc_low, rdu_low, rdc_low, ruu_medium, ruc_medium, rdu_medium, rdc_medium, ruu_high, ruc_high, rdu_high, rdc_high])**4).mean(0).sum()**(1/4)

    os.makedirs(args.output_directory + f'/undeployed/uncorrected/{idx}', exist_ok=True)
    os.makedirs(args.output_directory + f'/undeployed/corrected/{idx}', exist_ok=True)
    os.makedirs(args.output_directory + f'/deployed/uncorrected/{idx}', exist_ok=True)
    os.makedirs(args.output_directory + f'/deployed/corrected/{idx}', exist_ok=True)

    immono(anomaly_boundary(data2img(xu), seg).detach().numpy(), filename=args.output_directory + f'/{idx}_undeployed.png')
    immono(anomaly_boundary(data2img(xd), seg).detach().numpy(), filename=args.output_directory + f'/{idx}_deployed.png')

    immono(ruu_low / norm, filename=args.output_directory + f'/undeployed/uncorrected/{idx}/low.png', vmin=vmin, vmax=vmax)
    immono(ruc_low / norm, filename=args.output_directory + f'/undeployed/corrected/{idx}/low.png', vmin=vmin, vmax=vmax)
    immono(rdu_low / norm, filename=args.output_directory + f'/deployed/uncorrected/{idx}/low.png', vmin=vmin, vmax=vmax)
    immono(rdc_low / norm, filename=args.output_directory + f'/deployed/corrected/{idx}/low.png', vmin=vmin, vmax=vmax)

    immono(ruu_medium / norm, filename=args.output_directory + f'/undeployed/uncorrected/{idx}/medium.png', vmin=vmin, vmax=vmax)
    immono(ruc_medium / norm, filename=args.output_directory + f'/undeployed/corrected/{idx}/medium.png', vmin=vmin, vmax=vmax)
    immono(rdu_medium / norm, filename=args.output_directory + f'/deployed/uncorrected/{idx}/medium.png', vmin=vmin, vmax=vmax)
    immono(rdc_medium / norm, filename=args.output_directory + f'/deployed/corrected/{idx}/medium.png', vmin=vmin, vmax=vmax)

    immono(ruu_high / norm, filename=args.output_directory + f'/undeployed/uncorrected/{idx}/high.png', vmin=vmin, vmax=vmax)
    immono(ruc_high / norm, filename=args.output_directory + f'/undeployed/corrected/{idx}/high.png', vmin=vmin, vmax=vmax)
    immono(rdu_high / norm, filename=args.output_directory + f'/deployed/uncorrected/{idx}/high.png', vmin=vmin, vmax=vmax)
    immono(rdc_high / norm, filename=args.output_directory + f'/deployed/corrected/{idx}/high.png', vmin=vmin, vmax=vmax)
