import matplotlib.pyplot as plt
import json

categories = ['bottle', 'capsule', 'pill', 'toothbrush', 'wood']
corruptions = ['blurring', 'sharpening', 'checkerboard', 'salt&pepper']
algorithms = ['d2neighbors', 'supervised', 'patchcore']
keys = ['original'] + [item for corruption in corruptions for item in (corruption, f'{corruption}_corrected')]


fig, axes = plt.subplots(nrows=len(categories), ncols=len(algorithms), figsize=(18, 10))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
plt.suptitle('Performance Comparison per Algorithm and Dataset', fontsize=16)

for j, algorithm in enumerate(algorithms):
    with open(f'results/scores_with_artifacts/scores_{algorithm}.json', 'r') as f:
        data = json.load(f)

    for i, category in enumerate(categories):
        ax = axes[i][j]
        scores = [data[category].get(key, 0) for key in keys]
        bar_colors = ['#1f77b4' if 'corrected' not in key else '#ff7f0e' for key in keys]
        ax.bar(keys, scores, color=bar_colors)
        if i == len(categories) - 1:
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels([key.replace('_', ' ') for key in keys], rotation=45, ha="right")
        else:
            ax.set_xticks([])
        ax.set_ylim([0, 1])
        ax.grid(True)

for ax, cat in zip(axes[:, 0], categories):
    ax.set_ylabel(cat.capitalize())

for ax, alg in zip(axes[0], algorithms):
    ax.set_title(alg)

plt.tight_layout()
plt.savefig('results/scores_with_artifacts/scores_comparison.png')
