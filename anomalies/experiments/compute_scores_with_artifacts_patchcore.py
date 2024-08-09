from argparse import ArgumentParser
args = ArgumentParser()
args.add_argument('--category', type=str)
args.add_argument('--artifact', type=str, default='original')
args.add_argument('--corrected', type=str, default='False')

args = args.parse_args()
if args.corrected == 'False':
    args.corrected = False
elif args.corrected == 'True':
    args.corrected = True

print(args)

from src.models import PatchCore
from src.data import load_data_tensor, CATEGORIES, transform
from src.metrics import optimal_threshold, instance_metric
from src.utils import immono, data2img, anomaly_boundary
from src.artifacts import artifacts
from src.correction import Correction_circ as Correction
import torch as tr
import torch
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
import json
import sys

from src.utils import immono, data2img

correction = Correction(low=2, high=20)

path = f'results/scores_with_artifacts'
if not os.path.exists(path):
    os.makedirs(path)

common_args = {
    'n': 32,
    'ref': None,
    'kernel_size': (31, 31),
    'sigma': 1.0,
}

print('loading data...')
Xtrain, Xtest, ytest, Seg = load_data_tensor(args.category, transform=transform, target_transform=transform)

if args.artifact == 'original':
    pass
else:
    mean = Xtrain.mean(0)
    common_args['ref'] = mean
    artifact = artifacts[args.artifact]
    Xtest_clean = Xtest.clone()
    Xtest = tr.stack([artifact(x, **common_args) for x in Xtest])
    # stack the clean images with the artifacts
    Xtest = tr.cat([Xtest_clean, Xtest])

if args.corrected:
    Xtrain = correction(Xtrain)
    Xtest = tr.stack([correction(x.unsqueeze(0)).squeeze() for x in Xtest])

print('loading model...')
backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
patchcore = PatchCore(backbone=backbone)
patchcore.fit(Xtrain.unsqueeze(1))

print('computing scores...')
y_score = np.array([patchcore(x.unsqueeze(0))[0].item() for x in Xtest])
if args.artifact == 'original':
    threshold = optimal_threshold(ytest.numpy(), y_score)
else:
    y_score_clean = y_score[:len(Xtest_clean)]
    y_score = y_score[len(Xtest_clean):]
    threshold = optimal_threshold(ytest.numpy(), y_score_clean)
ypred = torch.from_numpy(y_score > threshold)
score = instance_metric(ytest.numpy(), ypred)
# scores[args.category][the_key] = score
print(f'patchcore balanced f1 score: %.2f'%(100*score))

print('reading and updating scores...')
the_key = f'{args.artifact}' if not args.corrected else f'{args.artifact}_corrected'

try:
    with open(f'{path}/scores_patchcore.json', 'r') as f:
        scores = json.load(f)
        if args.category not in scores:
            scores[args.category] = {}
        scores[args.category][the_key] = score
except FileNotFoundError:
    scores = {args.category: {the_key: score}}

print('writing scores to json file...')
with open(f'{path}/scores_patchcore.json', 'w') as f:
    json.dump(scores, f)

print('done, exiting...')

import os
import signal

pid = os.getpid()
os.kill(pid, signal.SIGKILL)
