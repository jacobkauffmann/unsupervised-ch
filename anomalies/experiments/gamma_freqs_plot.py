from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--scores', type=str, default='results/gamma_freqs/scores.json')
parser.add_argument('--model', type=str, default='D2Neighbors')
parser.add_argument('--output', type=str, default='results/gamma_freqs/plot.png')
# parser.add_argument('--category', type=str, default='toothbrush')
# parser.add_argument('--relevances', action='append', nargs='+')
args = parser.parse_args()

from dodo import gammas, combis
from src.correction import normalize_spectrum

import numpy as np
import torch as tr
import json

import matplotlib.pyplot as plt

with open(args.scores) as f:
    scores = json.load(f)

linestyles = {
    'cv2_resize': '-',
    'dither_quantize': '--',
    'none': ':'
}

colors = {
    model: {
        gamma: cm((i+1) / len(gammas))
        for i, gamma in enumerate(gammas)
    }
    for model, cm in zip(['D2Neighbors', 'Supervised'], [plt.cm.Reds, plt.cm.Blues])
}

linewidths = {
    'artifact': 1,
    'deployed': 2,
    'corrected': 3
}

scores = {key: value for key, value in sorted(scores.items())}

# we want three bar plots:
# 'none' artifact is the baseline and gets a bar for each model and gamma
# 'cv2_resize' artifact gets a bar for each model and gamma
# 'corrected' condition gets a bar for each model and gamma
model = args.model
max_score = 0
plt.figure(figsize=(10, 3))
for i, category in enumerate(scores):
    for artifact in scores[category][model]:
        for j, gamma in enumerate(scores[category][model][artifact]):
            if gamma != '1.0':
                continue
            for condition in ['fnr_artifact', 'fnr_deployed', 'fnr_corrected']:
                if artifact == 'none' and condition == 'fnr_artifact':
                    plt.subplot(131)
                elif artifact == 'cv2_resize' and condition == 'fnr_artifact':
                    plt.subplot(132)
                elif artifact == 'cv2_resize' and condition == 'fnr_corrected':
                    plt.subplot(133)
                else:
                    continue

                score = scores[category][model][artifact][gamma][condition]
                # use j to shift the bars
                plt.bar(i, score, width=.5)
                if score > max_score:
                    max_score = score

max_score = max_score * 1.1
plt.subplot(131)
plt.title('train+test w/ resize artifact')
plt.xticks(range(len(scores)), scores.keys(), rotation=45, ha='right')
plt.ylabel('balanced FNR')
plt.ylim(0, 1)
plt.subplot(132)
plt.title('tested on clean data')
plt.xticks(range(len(scores)), scores.keys(), rotation=45, ha='right')
plt.ylim(0, 1)
plt.yticks(plt.gca().get_yticks(), [])
plt.subplot(133)
plt.title('corrected model on clean data')
plt.xticks(range(len(scores)), scores.keys(), rotation=45, ha='right')
plt.ylim(0, 1)
plt.yticks(plt.gca().get_yticks(), [])

plt.tight_layout()
plt.savefig(args.output)
plt.close()

# plot the three confusion matrices in one plot
# they are in 'condition' = 'cm_artifact', 'cm_deployed', 'cm_corrected'
for i, category in enumerate(scores):
    plt.figure(figsize=(10, 3))
    for artifact in scores[category][model]:
        for j, gamma in enumerate(scores[category][model][artifact]):
            if gamma != '1.0':
                continue
            for condition in ['cm_evaluation', 'cm_deployed', 'cm_corrected']:
                if artifact == 'cv2_resize' and condition == 'cm_evaluation':
                    plt.subplot(131)
                elif artifact == 'cv2_resize' and condition == 'cm_deployed':
                    plt.subplot(132)
                elif artifact == 'cv2_resize' and condition == 'cm_corrected':
                    plt.subplot(133)
                else:
                    continue

                cm = scores[category][model][artifact][gamma][condition]
                plt.imshow(cm, cmap='Blues', vmin=0)
                # plt.colorbar()
                plt.title(condition)
                plt.xticks([0, 1], ['predicted 0', 'predicted 1'])
                plt.yticks([0, 1], ['true 0', 'true 1'])
                for y in range(2):
                    for x in range(2):
                        plt.text(x, y, f'{cm[y][x]}', ha='center', va='center', color='black')


    plt.tight_layout()
    plt.savefig(args.output.replace('.png', f'_{category}_cm.png'))
    plt.close()

# plt.figure(figsize=(10, 10))
# plt.subplot(211)
# for model in scores:
#     for artifact in scores[model]:
#         gammas = sorted(scores[model][artifact].keys())
#         for condition in ['artifact', 'deployed', 'corrected']:
#             scores_ = [scores[model][artifact][gamma][condition] for gamma in gammas]
#             plt.plot(np.log10([float(g) for g in gammas]), scores_, linestyle=linestyles[artifact], color=colors[model][float(gammas[-1])], linewidth=linewidths[condition])
# plt.xlabel('log10 gamma')
# plt.ylabel('F1 score')
# plt.subplot(212)
# for rel in args.relevances[0]:
#     index = int(rel.split('/')[-1].split('.')[0])
#     model, _, artifact, gamma = combis[index]
#     RR = tr.load(rel)
#     for condition in ['artifact', 'corrected']:
#         R = RR[condition]
#         R = normalize_spectrum(R.abs())
#         plt.plot(R, label=f'{model} {artifact} {gamma}', alpha=.5, color=colors[model][float(gamma)], linestyle=linestyles[artifact], linewidth=linewidths[condition])
# plt.yscale('log')
# plt.xlabel('frequency')
# plt.ylabel('relevance')

# from matplotlib.lines import Line2D

# # Define custom legend entries
# legend_elements = [
#     Line2D([0], [0], color='white', lw=3, linestyle='--', label='linestyle (artifact)'),
#     Line2D([0], [0], color='black', lw=3, linestyle='--', label='dithering'),
#     Line2D([0], [0], color='black', lw=3, linestyle='-', label='resize'),
#     Line2D([0], [0], color='black', lw=3, linestyle=':', label='no artifact'),
#     Line2D([0], [0], color='white', lw=3, label='color (model)'),
#     Line2D([0], [0], color='red', lw=3, label='D2Neighbors'),
#     Line2D([0], [0], color='blue', lw=3, label='Supervised'),
#     Line2D([0], [0], color='white', lw=3, label='(shades are different gammas)'),
#     Line2D([0], [0], color='white', lw=3, label='linewidth (condition)'),
#     Line2D([0], [0], color='black', lw=1, label='test threshold'),
#     Line2D([0], [0], color='black', lw=2, label='new test threshold (top plot only)'),
#     Line2D([0], [0], color='black', lw=3, label='corrected + new threshold')
# ]

# # Add legend to the plot with three columns
# plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

# plt.tight_layout()
# plt.savefig(args.output)

# plt.figure(figsize=(10, 10))
# # plt.title(category)
# plt.subplot(211)
# for model in scores[category]:
#     for artifact in scores[category][model]:
#         gammas = list(scores[category][model][artifact].keys())
#         scores_ = list(scores[category][model][artifact].values())
#         # plt.plot(np.log10(gammas), scores_, label=f'{model} {artifact}')
#         the_artifact = artifact if artifact == 'cv2_resize' else 'pillow_resize'
#         plt.plot(np.log10(gammas), scores_, label=f'{model} {the_artifact}', linestyle=linestyles[artifact], color=colors[model][gammas[-1]])
# plt.legend()
# plt.xticks(np.log10(gammas), [f'{g:.0e}' for g in gammas])
# plt.xlabel('gamma')
# plt.ylabel('F1 score')

# plt.subplot(212)
# for model in ['D2Neighbors', 'Supervised']:
#         for artifact in artifacts:
#         for gamma in spectra[category][model][artifact]:
#             R = spectra[category][model][artifact][gamma]
#             plt.plot(R, label=f'{model} {artifact} {gamma}', alpha=.5, color=colors[model][gamma], linestyle=linestyles[artifact])
# plt.yscale('log')
# plt.xlabel('frequency')
# plt.ylabel('power')

# plt.tight_layout()
# plt.savefig(args.output)
