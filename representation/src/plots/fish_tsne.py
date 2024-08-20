import sys

sys.path.append('.')
import argparse
import os
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data import FISH

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='resources/features/fish/std/test')
parser.add_argument('--output-dir', default='resources/tsne/fish')
args = parser.parse_args()

dual = False

with open('resources/imagenet_class_index.json') as f:
    class_index = json.load(f)
class_dict = {item[0]: item[1] for item in class_index.values()}
class_names = [class_dict[wnid] for wnid in FISH]
print(class_names)

person_indicator = np.load('resources/person_indicator/fish_val.npy')

for file in os.listdir(args.input):
    arr = np.load(os.path.join(args.input, file))
    labels = arr['labels']
    embeddings = arr['embeddings']

    colors = ['red', 'green', 'orange', 'brown', 'purple', 'pink']

    X = TSNE(n_components=2, learning_rate='auto',
             init='random',
             metric='cosine',
             random_state=0).fit_transform(embeddings)

    # X = PCA(n_components=2, random_state=0).fit_transform(X_t)

    classes = ['barracouta', 'tench', 'coho', 'sturgeon', 'gar']
    # classes = ['goldfish', 'lionfish', 'eel', 'rock_beauty']

    if dual:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches((10, 6))

        c = np.zeros(len(embeddings))
        for idx, cls in enumerate(classes):
            c[labels == class_names.index(cls)] = idx + 1

        ax[0].scatter(*zip(*X[c == 0]), c='black', label='other', alpha=.05)
        for i, category in enumerate(classes):
            ax[0].scatter(*zip(*X[c == i + 1]), c=colors[i], label=category, alpha=.1)
        ax[0].axis('off')
        leg = ax[0].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))

        for lh in leg.legendHandles:
            lh.set_alpha(1)

        ax[1].scatter(*zip(*X[person_indicator == 0]), c='black', label='other', alpha=.2)
        ax[1].scatter(*zip(*X[person_indicator]), c='blue', label='humans', alpha=.2)
        ax[1].axis('off')
        leg = ax[1].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))

        for lh in leg.legendHandles:
            lh.set_alpha(1)
    else:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches((6, 6))

        ax.scatter(*zip(*X[person_indicator == 0]), c='black', label='other', alpha=.2)
        ax.scatter(*zip(*X[person_indicator]), c='blue', label='humans', alpha=.2)
        ax.axis('off')
        #ax.get_legend().remove()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, file.removesuffix('.npz') + f'.png'), bbox_inches='tight')
