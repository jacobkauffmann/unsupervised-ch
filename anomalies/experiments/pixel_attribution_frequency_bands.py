from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--category", type=str, required=True)
parser.add_argument("--artifact", type=str, required=True)
parser.add_argument("--low", type=int, required=True, default=2)
parser.add_argument("--high", type=int, required=True, default=20)
parser.add_argument("--corrected", action="store_true")
parser.add_argument("--deployed", action="store_true")
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

steps = [0, args.low, args.high, 50176]

from src.data import load_data_tensor, transform, target_transform
from src.utils import split_for_supervised
from src.artifacts import artifact_transform, artifacts
from src.models import models
from src.correction import correction_layer, Correction_circ

import torch as tr
import sys

if args.artifact == 'dither':
    train_transform = artifact_transform('dither', dither=False)
    deployed_transform = artifact_transform('dither', dither=True)
else:
    train_transform = artifact_transform(args.artifact)
    deployed_transform = artifact_transform('none')

Xtrain, Xtest, ytest, Seg = load_data_tensor(args.category, transform=train_transform, target_transform=target_transform)

_, Xtest_deployed, _, _ = load_data_tensor(args.category, transform=deployed_transform, target_transform=target_transform)

def explain_indices(x, low, high, model, eps=1e-6):
    myfilter = Correction_circ(low=low, high=high)
    x = x.detach().clone()
    model.svs.requires_grad_(True)
    model.svs.grad = None
    delta = x - model.svs
    output = model(x.unsqueeze(0)).squeeze()
    factor = output / ((delta.flatten(1)**model.p).sum(1) + eps)
    factor = factor.view(-1, *([1]*(delta.dim()-1)))
    output.backward()
    R = -(1/model.p)*(delta * myfilter(model.svs.grad) * factor).sum(0).reshape(x.shape)
    model.svs.requires_grad_(False)
    model.svs.detach_()
    # detach layer's parameters
    if isinstance(model.layer, tr.nn.Module):
        for param in model.layer.parameters():
            param.detach_()
    return R

if args.corrected:
    layer = correction_layer()
else:
    layer = lambda x: x

model = models[args.model](layer=layer)
if 'D2Neighbors' in args.model:
    model.fit(Xtrain)
else:
    # not implemented
    raise NotImplementedError

if args.deployed:
    Xtest = Xtest_deployed

Rlow, Rmedium, Rhigh = [], [], []
for index in range(len(Xtest)):
    for the_R, (low, high) in zip([Rlow, Rmedium, Rhigh], zip(steps[:-1], steps[1:])):
        R = explain_indices(Xtest[index], low, high, model).sum(0).detach()
        the_R.append(R)

Rlow, Rmedium, Rhigh = tr.stack(Rlow), tr.stack(Rmedium), tr.stack(Rhigh)

R = {"low": Rlow, "medium": Rmedium, "high": Rhigh}
tr.save(R, args.output)
