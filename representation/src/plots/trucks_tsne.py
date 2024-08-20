import sys
sys.path.append('.')
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='resources/features/trucks/watermark_half/train')
parser.add_argument('--output-dir', default='resources/tsne/trucks')
args = parser.parse_args()

indices = list(range(10259))
indices = random.sample(indices, k=3000)

for file in os.listdir(args.input):
    arr = np.load(os.path.join(args.input, file))
    embeddings = np.asarray(arr['embeddings'], dtype='float64')[indices]
    y = arr['labels'][indices]
    print(embeddings.shape)
    n_labels = len(np.unique(y))
    logo_labels = (y >= (n_labels / 2)).astype(np.int32)

    X = TSNE(n_components=2, learning_rate='auto',
             init='random',
             metric='cosine',
             random_state=0).fit_transform(embeddings)

    # X = PCA(n_components=2, svd_solver='full').fit_transform(embeddings)

    class_labels = np.zeros(y.shape).astype(np.float32) - 1
    assert n_labels == 16
    for idx in range(int(n_labels / 2)):
        class_labels[y == idx] = idx
        class_labels[y == (idx + int(n_labels / 2))] = idx

    plt.scatter(*zip(*X[logo_labels == 1]), c='blue', alpha=0.2, label='no logo')
    plt.scatter(*zip(*X[logo_labels == 0]), c='orange', alpha=0.2, label='logo')
    plt.axis('off')
    plt.legend().remove()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, file.removesuffix('.npz') + f'.png'), bbox_inches='tight')
    plt.clf()
