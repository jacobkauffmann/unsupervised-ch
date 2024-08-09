import matplotlib.pyplot as plt
import numpy as np
import json

models = ['D2Neighbors', 'PatchCore', 'D2NeighborsL1', 'D2NeighborsL4']
categories = ['bottle', 'capsule', 'pill', 'toothbrush', 'wood']
artifacts = ['cv2resize']

# pick a smooth set of three colors
colors = plt.cm.get_cmap('tab10', 5)
colors = colors([0, 1, 2, 3, 4])

# the plot above groups the bars by category, but we can also group them by condition (evaluation, deployed, corrected)
bar_width = 0.15
alpha_val = 0.5
for model in models:
    for artifact in artifacts:
        plt.figure(figsize=(4, 1.5))
        for idx, condition in enumerate(['cm_evaluation', 'cm_deployed', 'cm_corrected']):
            vals = []
            for jdx, category in enumerate(categories):
                with open(f'results/scores/{model}/{artifact}/{category}.json') as f:
                    # score = json.load(f)[condition]
                    cm = json.load(f)[condition]
                    fnr = cm[1][0] / (cm[1][0] + cm[1][1])
                    score = fnr
                x_pos = idx + jdx*bar_width - 2*bar_width
                plt.bar(x_pos, score, width=bar_width, color=colors[jdx], label=category, alpha=alpha_val)
                vals += [score]
            mean_val = np.mean(vals)
            plt.plot([idx-3*bar_width, idx+3*bar_width], [mean_val, mean_val], color='black', linestyle='-', linewidth=2)
        if True:#model == 'PatchCore':
            plt.xticks(range(3), ['evaluation', 'deployed', 'corrected'])
        else:
            plt.xticks(range(3), [])
        if True:#artifact == 'dither':
            plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ['0.0', '', '0.5', '', '1.0'])
        else:
            plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], [])
        plt.ylim(0, 1)
        plt.subplots_adjust(left=0.1, right=.98, top=.95, bottom=0.18)
        # plt.tight_layout()
        plt.savefig(f'results/scores/{model}/fnr_{artifact}.pdf', dpi=300)
        plt.close()

# plot the legend as separate file
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none", alpha=alpha_val)[0]
labels = ['bottle', 'capsule', 'pill', 'toothbrush', 'wood']
handles = [f("s", colors[i]) for i in range(len(labels))]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)

def export_legend(legend, filename="results/scores/legend.pdf", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
