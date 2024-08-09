from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--model', type=str, default='D2Neighbors')
parser.add_argument('--category', type=str, default='toothbrush')
parser.add_argument('--artifact', type=str, default='cv2_resize')
parser.add_argument('--imsize', type=int, default=64)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--output', type=str, default='results/gamma_freqs/scores.json')
args = parser.parse_args()

# overwrite gamma with 0.001
# args.gamma = 0.001

from src.models import models
from src.data import load_data_tensor, CATEGORIES, transform, target_transform
from src.artifacts import artifact_transform
from src.utils import split_for_supervised, immono, data2img
from src.metrics import instance_metric, optimal_threshold
from src.correction import power_spectrum, normalize_spectrum, data2freq, freq2data
from src.correction import correction_layer

import torch as tr
# device = tr.device('mps')
device = tr.device('mps') if tr.backends.mps.is_available() else (tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu'))
# device = tr.device('cpu')

import sys
import matplotlib.pyplot as plt
import numpy as np
import json

# data2freq_ = ApplyPerChannel(V=V).to(device)
# def data2freq(x):
#     x = data2freq_(x)
#     x = x.view(-1, 3, 64, 64)
#     return x
# freq2data_ = ApplyPerChannel(V=V.T).to(device)
# def freq2data(z):
#     z = freq2data_(z)
#     z = z.view(z.shape[0], 3, 64, 64)
#     # z = z.view(z.shape[0], -1)
#     return z

if args.artifact == 'dither':
    train_transform = artifact_transform('quantize')
    deployed_transform = artifact_transform('dither')
else:
    train_transform = artifact_transform(args.artifact)
    deployed_transform = artifact_transform('none')

Xtrain, Xtest, ytest, Seg = load_data_tensor(args.category, transform=train_transform, target_transform=target_transform)

_, Xtest_deployed, _, _ = load_data_tensor(args.category, transform=deployed_transform, target_transform=target_transform)

Xtrain, Xtest, Xtest_deployed = Xtrain.to(device), Xtest.to(device), Xtest_deployed.to(device)

Xtrain, Xtest, Xtest_deployed = data2freq(Xtrain), data2freq(Xtest), data2freq(Xtest_deployed)

model = models[args.model](layer=freq2data)
model_corrected = models[args.model](layer=correction_layer(freq2data))

if 'D2Neighbors' in args.model:
    model.fit(Xtrain)

    model_corrected.fit(Xtrain)
else:
    Xtrain_, Xtest_, ytrain, _ = split_for_supervised(Xtrain, Xtest, ytest, test_size=0.8, random_state=42)
    _, Xtest_deployed, _, ytest = split_for_supervised(Xtrain, Xtest_deployed, ytest, test_size=0.8, random_state=42)
    Xtrain, Xtest = Xtrain_, Xtest_
    model.fit(Xtrain, ytrain)
    model.gamma = args.gamma
    # m = model.explain
    # model.explain = lambda x: m(x, 1)

    model_corrected.fit(Xtrain, ytrain)
    model_corrected.gamma = args.gamma
    # m_corrected = model_corrected.explain
    # model_corrected.explain = lambda x: m_corrected(x, 1)

R_undeployed_uncorrected = []
R_undeployed_corrected = []
R_deployed_uncorrected = []
R_deployed_corrected = []
model.to(device)
model_corrected.to(device)
for i in range(len(Xtest)):
    r = model.explain(Xtest[i].unsqueeze(0)).data.cpu()
    r = r.reshape(3,args.imsize,args.imsize).sum(0).flatten()
    R_undeployed_uncorrected.append(power_spectrum(r))

    r = model_corrected.explain(Xtest_deployed[i].unsqueeze(0)).data.cpu()
    r = r.reshape(3,args.imsize,args.imsize).sum(0).flatten()
    R_deployed_corrected.append(power_spectrum(r))

    r = model.explain(Xtest_deployed[i].unsqueeze(0)).data.cpu()
    r = r.reshape(3,args.imsize,args.imsize).sum(0).flatten()
    R_deployed_uncorrected.append(power_spectrum(r))

    r = model_corrected.explain(Xtest[i].unsqueeze(0)).data.cpu()
    r = r.reshape(3,args.imsize,args.imsize).sum(0).flatten()
    R_undeployed_corrected.append(power_spectrum(r))
R_undeployed_uncorrected = tr.stack(R_undeployed_uncorrected)
R_undeployed_corrected = tr.stack(R_undeployed_corrected)
R_deployed_uncorrected = tr.stack(R_deployed_uncorrected)
R_deployed_corrected = tr.stack(R_deployed_corrected)

relevance = {
    'undeployed': {
        'uncorrected': R_undeployed_uncorrected,
        'corrected': R_undeployed_corrected
    },
    'deployed': {
        'uncorrected': R_deployed_uncorrected,
        'corrected': R_deployed_corrected
    }
}

# write R to pt file
tr.save(relevance, args.output)
