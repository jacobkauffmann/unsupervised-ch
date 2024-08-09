import numpy as np
import matplotlib.pyplot as plt
import json
import os

# load scores from json
with open('results/scores/scores.json', 'r') as f:
    scores = json.load(f)

path = 'results/scores'

# structure is scores[category][beta]
# we want to plot the scores for each beta
categories = list(scores.keys())
metrics = list(scores[categories[0]].keys())

for metric in metrics:
    scores_metric = [scores[category][metric] for category in categories]
    # sort from highest to lowest
    categories, scores_metric = zip(*sorted(zip(categories, scores_metric), key=lambda x: x[1], reverse=True))
    plt.figure()
    plt.bar(categories, scores_metric, zorder=3)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{metric} score')
    plt.ylabel('score')
    plt.xlabel('category')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', zorder=0)
    plt.tight_layout()
    plt.savefig(f'{path}/{metric}_score.png')
