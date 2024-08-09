from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--model', type=str, default='D2Neighbors')
parser.add_argument('--category', type=str, default='toothbrush')
parser.add_argument('--artifact', type=str, default='cv2_resize')
parser.add_argument('--imsize', type=int, default=64)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--output', type=str, default='results/gamma_freqs/0.json')
args = parser.parse_args()
print(args)

from src.models import models
from src.data import load_data_tensor, CATEGORIES, transform, target_transform
from src.artifacts import artifact_transform
from src.utils import split_for_supervised, immono, data2img
from src.metrics import instance_metric, optimal_threshold, false_negative_rate
from src.correction import power_spectrum, normalize_spectrum, V
from src.correction import correction_layer

import torch as tr
device = tr.device('cpu')

import sys
import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.metrics import confusion_matrix

# data2freq_ = ApplyPerChannel(V=V).to(device)
# def data2freq(x):
#     x = data2freq_(x)
#     x = x.view(-1, 3, 64, 64)
#     return x
# freq2data_ = ApplyPerChannel(V=V.T).to(device)
# def freq2data(z):
#     z = freq2data_(z)
#     z = z.view(z.shape[0], -1)
#     return z

if args.artifact == 'dither':
    train_transform = artifact_transform('quantize')
    deployed_transform = artifact_transform('dither')
elif args.artifact == 'gaussiannoise':
    train_transform = artifact_transform('none')
    deployed_transform = artifact_transform('gaussiannoise')
else:
    train_transform = artifact_transform(args.artifact)
    deployed_transform = artifact_transform('none')

train_transform = artifact_transform(args.artifact)
Xtrain, Xtest, ytest, Seg = load_data_tensor(args.category, transform=train_transform, target_transform=target_transform)

_, Xtest_deployed, _, _ = load_data_tensor(args.category, transform=deployed_transform, target_transform=target_transform)

Xtrain, Xtest, Xtest_deployed = Xtrain.to(device), Xtest.to(device), Xtest_deployed.to(device)

# Xtrain, Xtest, Xtest_deployed = data2freq(Xtrain), data2freq(Xtest), data2freq(Xtest_deployed)

# model = models[args.model](layer=freq2data)
# model_corrected = models[args.model](layer=correction_layer(freq2data))
model = models[args.model](imsize=(args.imsize, args.imsize))
model_corrected = models[args.model](layer=correction_layer(), imsize=(args.imsize, args.imsize))

if args.model != 'MLP':
    model.fit(Xtrain)

    model_corrected.fit(Xtrain)
else:
    Xtrain_, Xtest_, ytrain, _ = split_for_supervised(Xtrain, Xtest, ytest, test_size=0.8, random_state=42)
    _, Xtest_deployed, _, ytest = split_for_supervised(Xtrain, Xtest_deployed, ytest, test_size=0.8, random_state=42)
    Xtrain, Xtest = Xtrain_, Xtest_
    model.fit(Xtrain, ytrain)
    # model.gamma = args.gamma
    m = model.forward
    model.forward = lambda x: m(x).sum((1,2,3))

    model_corrected.fit(Xtrain, ytrain)
    # model_corrected.gamma = args.gamma
    m_corrected = model_corrected.forward
    model_corrected.forward = lambda x: m_corrected(x).sum((1,2,3))

ypred_evaluation = model(Xtest).detach().cpu()
threshold = optimal_threshold(ytest, ypred_evaluation)
ypred_deployed = model(Xtest_deployed).detach().cpu()
threshold_deployed = optimal_threshold(ytest, ypred_deployed)
# ypred_corrected = model_corrected(Xtest_deployed).detach().cpu()
ypred_corrected_evaluation = model_corrected(Xtest).detach().cpu()
threshold_corrected = optimal_threshold(ytest, ypred_corrected_evaluation)
ypred_corrected_deployed = model_corrected(Xtest_deployed).detach().cpu()

score_evaluation = instance_metric(ytest, ypred_evaluation > threshold)
score_deployed = instance_metric(ytest, ypred_deployed > threshold)
score_corrected = instance_metric(ytest, ypred_corrected_deployed > threshold_corrected)

# also track the false-negative rates
fnr_artifact = false_negative_rate(ytest, ypred_deployed > threshold)
fnr_deployed = false_negative_rate(ytest, ypred_deployed > threshold_deployed)

fnr_corrected = false_negative_rate(ytest, ypred_corrected_deployed > threshold_corrected)

# additonally, we decided to store the confusion matrices as well
cm_evaluation = confusion_matrix(ytest, ypred_evaluation > threshold).tolist()
# cm_deployed = confusion_matrix(ytest, ypred_deployed > threshold_deployed).tolist()
cm_deployed = confusion_matrix(ytest, ypred_deployed > threshold).tolist()
cm_corrected = confusion_matrix(ytest, ypred_corrected_deployed > threshold_corrected).tolist()
cm_corrected_evaluation = confusion_matrix(ytest, ypred_corrected_evaluation > threshold_corrected).tolist()

# now a cm has the following structure:
# [[true negatives, false positives],
#  [false negatives, true positives]]

score = {
    'f1_evaluation': score_evaluation,
    'f1_deployed': score_deployed,
    'f1_corrected': score_corrected,
    'fnr_artifact': fnr_artifact,
    'fnr_deployed': fnr_deployed,
    'fnr_corrected': fnr_corrected,
    # 'cm_artifact': cm_artifact,
    'cm_evaluation': cm_evaluation,
    'cm_deployed': cm_deployed,
    'cm_corrected': cm_corrected,
    'cm_corrected_evaluation': cm_corrected_evaluation
}

with open(args.output, 'w') as f:
    json.dump(score, f)

### everything below does not use the arguments
### we want to implement a simple version of the code below
### without any loops
# scores = {}
# spectra = {}

# for category in CATEGORIES:
#     scores[category] = {}
#     spectra[category] = {}
#     Xtrain, Xtest, ytest, Seg = load_data_tensor(category, transform=transform, target_transform=transform)
#     _, Xtest_artifact, _, _ = load_data_tensor(category, transform=transform_cv2, target_transform=transform_cv2)
#     # _, Xtest_artifact, _, _ = load_data_tensor(category, transform=transform_cv2, target_transform=transform_gaussian_noise)
#     Xtrain, Xtest, Xtest_artifact = Xtrain.to(device), Xtest.to(device), Xtest_artifact.to(device)
#     Xtrain, Xtest, Xtest_artifact = data2freq(Xtrain), data2freq(Xtest), data2freq(Xtest_artifact)

#     # D2Neighbors
#     scores[category]['D2Neighbors'] = {}
#     spectra[category]['D2Neighbors'] = {}
#     model = MahalanobisKDE(layer=freq2data)
#     model.fit(Xtrain)
#     for artifact in artifacts:
#         scores[category]['D2Neighbors'][artifact] = {}
#         spectra[category]['D2Neighbors'][artifact] = {}
#         for gamma in gammas:
#             model.gamma = gamma
#             ypred_ = model(Xtest).detach().cpu().numpy()
#             threshold = optimal_threshold(ytest, ypred_)
#             # ypred = model(artifacts[artifact](Xtest)).detach()
#             if artifact == 'none':
#                 ypred = model(Xtest).detach().cpu()
#             else:
#                 ypred = model(Xtest_artifact).detach().cpu()
#             print(f'{category} {artifact} {gamma} D2Neighbors', instance_metric(ytest, ypred > threshold))
#             scores[category]['D2Neighbors'][artifact][gamma] = instance_metric(ytest, ypred > threshold)
#             RR = []
#             for x in Xtest_artifact[(ytest == 1) & (ypred > threshold)]:
#                 x = x.unsqueeze(0)
#                 R = model.explain(x).detach().cpu()
#                 R = R.reshape(3,64,64).sum(0).flatten()
#                 R = power_spectrum(R)
#                 RR.append(R)
#             R = normalize_spectrum(tr.stack(RR)).squeeze()
#             spectra[category]['D2Neighbors'][artifact][gamma] = R


#     # Supervised
#     scores[category]['Supervised'] = {}
#     spectra[category]['Supervised'] = {}
#     model = Supervised(layer=freq2data)
#     Xtrain_, Xtest_, ytrain_, ytest_ = split_for_supervised(Xtrain, Xtest, ytest, test_size=0.8, random_state=42)
#     _, Xtest_artifact_, _, _ = split_for_supervised(Xtrain, Xtest_artifact, ytest, test_size=0.8, random_state=42)
#     Xtrain_, Xtest_, Xtest_artifact_ = Xtrain_.to(device), Xtest_.to(device), Xtest_artifact_.to(device)
#     # from sklearn.cluster import KMeans
#     # Xtrain_ = tr.stack([
#     #     tr.from_numpy(KMeans(n_clusters=10).fit(Xtrain_[ytrain_ == c].reshape(-1,12288).detach().cpu().numpy()).cluster_centers_) for c in tr.unique(ytrain_)
#     # ]).reshape(-1,3,64,64).float().to(device)
#     # ytrain_ = tr.cat([tr.full((10,), c) for c in tr.unique(ytrain_)]).to(device)
#     # NOTE: data is already in frequency domain!!!
#     # Xtrain_, Xtest_, Xtest_artifact_ = data2freq(Xtrain_), data2freq(Xtest_), data2freq(Xtest_artifact_)

#     model.fit(Xtrain_, ytrain_)
#     for artifact in artifacts:
#         scores[category]['Supervised'][artifact] = {}
#         spectra[category]['Supervised'][artifact] = {}
#         for gamma in gammas:
#             model.gamma = gamma
#             ypred_ = model(Xtest_)[:,1].detach().cpu()
#             threshold = optimal_threshold(ytest_, ypred_)
#             if artifact == 'none':
#                 ypred = model(Xtest_)[:,1].detach().cpu()
#             else:
#                 ypred = model(Xtest_artifact_)[:,1].detach().cpu()
#             print(f'{category} {artifact} {gamma} Supervised', instance_metric(ytest_, ypred > threshold))
#             scores[category]['Supervised'][artifact][gamma] = instance_metric(ytest_, ypred > threshold)
#             RR = []
#             for x in Xtest_artifact_[(ytest_ == 1) & (ypred > threshold)]:
#                 x = x.unsqueeze(0)
#                 R = model.explain(x, 1).detach().cpu()
#                 R = R.reshape(3,64,64).sum(0).flatten()
#                 R = power_spectrum(R)
#                 RR.append(R)
#             # NOTE: we take an abs here just for visualization purposes
#             R = normalize_spectrum(tr.stack(RR).abs()).squeeze()
#             spectra[category]['Supervised'][artifact][gamma] = R

# linestyles = {
#     'cv2_resize': '-',
#     'none': ':'
# }

# colors = {
#     model: {
#         gamma: cm(i / len(gammas))
#         for i, gamma in enumerate(gammas)
#     }
#     for model, cm in zip(['D2Neighbors', 'Supervised'], [plt.cm.Reds, plt.cm.Blues])
# }

# for category in CATEGORIES:
#     plt.figure(figsize=(10, 10))
#     # plt.title(category)
#     plt.subplot(211)
#     for model in scores[category]:
#         for artifact in scores[category][model]:
#             gammas = list(scores[category][model][artifact].keys())
#             scores_ = list(scores[category][model][artifact].values())
#             # plt.plot(np.log10(gammas), scores_, label=f'{model} {artifact}')
#             the_artifact = artifact if artifact == 'cv2_resize' else 'pillow_resize'
#             plt.plot(np.log10(gammas), scores_, label=f'{model} {the_artifact}', linestyle=linestyles[artifact], color=colors[model][gammas[-1]])
#     plt.legend()
#     plt.xticks(np.log10(gammas), [f'{g:.0e}' for g in gammas])
#     plt.xlabel('gamma')
#     plt.ylabel('F1 score')

#     plt.subplot(212)
#     for model in ['D2Neighbors', 'Supervised']:
#          for artifact in artifacts:
#             for gamma in spectra[category][model][artifact]:
#                 R = spectra[category][model][artifact][gamma]
#                 plt.plot(R, label=f'{model} {artifact} {gamma}', alpha=.5, color=colors[model][gamma], linestyle=linestyles[artifact])
#     plt.yscale('log')
#     plt.xlabel('frequency')
#     plt.ylabel('power')

#     plt.tight_layout()
#     plt.savefig(f'results/gamma_freqs/{category}.png')

# # save scores
# import json
# with open('results/gamma_freqs/scores.json', 'w') as f:
#     json.dump(scores, f)
