from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--category', type=str, default='toothbrush')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--model', type=str, default='D2Neighbors')
parser.add_argument('--artifact', type=str, default='none')
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--corrected', action='store_true')
parser.add_argument('--deployed', action='store_true')
parser.add_argument('--output', type=str)

args = parser.parse_args()
# output = f'results/figure_explanations/{args.category}_%d_{args.artifact}_{args.gamma}_' + ('corrected' if args.corrected else 'uncorrected') + '_' + ('deployed' if args.deployed else 'undeployed') + '_%s.png'
print(args)

from src.data import load_data_tensor, transform, target_transform
from src.utils import split_for_supervised, immono, data2img, anomaly_boundary
from src.artifacts import artifacts, artifact_transform
from src.models import models
from src.correction import correction_layer

import torch as tr
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

if args.artifact == 'dither':
    train_transform = artifact_transform('quantize')
    deployed_transform = artifact_transform('dither')
else:
    train_transform = artifact_transform(args.artifact)
    deployed_transform = artifact_transform('none')

Xtrain, Xtest, ytest, Seg = load_data_tensor(args.category, transform=train_transform, target_transform=target_transform)

_, Xtest_deployed, _, _ = load_data_tensor(args.category, transform=deployed_transform, target_transform=target_transform)

if args.corrected:
    layer = correction_layer()
else:
    layer = lambda x: x

model = models[args.model](layer=layer)
if 'D2Neighbors' in args.model:
    model.fit(Xtrain)
else:
    Xtrain_, Xtest_, ytrain, _ = split_for_supervised(Xtrain, Xtest, ytest, test_size=0.8, random_state=42)
    _, Xtest_deployed, _, ytest = split_for_supervised(Xtrain, Xtest_deployed, ytest, test_size=0.8, random_state=42)
    Xtrain, Xtest = Xtrain_, Xtest_
    model.fit(Xtrain, ytrain)
    m = model.forward
    model.forward = lambda x: m(x).sum((1,2))

if args.deployed:
    Xtest = Xtest_deployed

# first iteration: collect all explanations
RR = []
for index in tqdm(range(min(args.index, len(Xtest)))):
    x = Xtest[index].unsqueeze(0).detach().clone()
    # seg = Seg[index].detach().clone()

    R = model.explain(x).sum(0).detach()
    RR += [R]
RR = tr.stack(RR)

# global_norm = (RR**4).mean(0).sum()**(1/4)
# # second iteration: plot all explanations w/ global normalization
# for index, R in enumerate(RR):
#     x = Xtest[index].detach().clone()
#     seg = Seg[index].detach().clone()
#     # normalize heatmap by 4-norm
#     R = R / global_norm
#     R = R.detach().numpy()

#     immono(anomaly_boundary(data2img(x), seg).detach().numpy(), filename=output % (index, 'input'))
#     immono(R, filename=output % (index, 'explanation'), vmin=-1, vmax=1)

# actually, we need global normalization over all condititions.
# so we write the explanations to disk and then load them in another script

tr.save(RR, args.output)
