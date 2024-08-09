from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--category', type=str, default='toothbrush')
parser.add_argument('--artifact', type=str, default='none')
parser.add_argument('--frequency-file', type=str)
# parser.add_argument('--second_model', type=str)
parser.add_argument('--model', type=str, default='D2Neighbors')
# parser.add_argument('--model2', type=str, default='MLP')
parser.add_argument('--output_directory', type=str)
parser.add_argument('--low', type=int, default=2)
parser.add_argument('--high', type=int, default=20)
args = parser.parse_args()

import sys

from dodo import combis
from src.correction import normalize_spectrum, power_spectrum

import numpy as np
import torch as tr
import matplotlib.pyplot as plt

linestyles = {
    'uncorrected': '-',
    # 'MLP': '--'
    'corrected': '--'
}

colors = {
    'uncorrected': '#DD8888',##1f77b4',#'blue',
    'corrected': '#DD8888'
    # 'MLP': 'green'
}

# alphas = {
#     'uncorrected': 1,
#     'corrected': 0.3
# }

def sum_over_range(data, start, end):
    return np.sum(data[start:end + 1])

p = 3.47
def get_bins(p=3.47):
    bins = np.linspace(0, (224*224)**(1/p), 21)**p
    return bins.round().astype(int)
bins = get_bins()
freq_ranges = [(bins[0], bins[1])] + [(bins[i] + 1, bins[i + 1]) for i in range(1, len(bins) - 1)]

colors = {
    'low': '#DD8888', # '#804f4f',
    'mid': '#DD8888',
    'high': '#DD8888' # '#FFDDDD'
}

ylim = [float('inf'), -float('inf')]
figures = []

for deployed in ['undeployed', 'deployed']:
    for corrected in ['uncorrected', 'corrected']:
    #     fig, ax = plt.subplots(figsize=(3, 2))
    #     for relevance_file, model_name in [(args.first_model, args.model1), (args.second_model, args.model2)]:
    #         full_data = tr.load(relevance_file)
    #         data = full_data[deployed][corrected]
    #         data = normalize_spectrum(data.abs())
    #         data = data.detach().numpy()
    #         ylim = [min(ylim[0], data.min()), max(ylim[1], data.max())]
    #         ax.plot(data, label=corrected, color=colors[model_name], linestyle=linestyles[model_name])
    #     figures += [(fig, ax, deployed, corrected)]
        fig, ax = plt.subplots(figsize=(3, 2))
        # for relevance_file, model_name, corrected in [(args.first_model, args.model1, 'uncorrected'), (args.second_model, args.model2, 'corrected')]:
        for relevance_file, model_name in [(args.frequency_file , args.model)]:
            full_data = tr.load(relevance_file)
            data = full_data[deployed][corrected]
            data = normalize_spectrum(data)
            data = data.detach().numpy()
            data = np.array([sum_over_range(data, start, end) for start, end in freq_ranges])
            ylim = [min(ylim[0], data.min()), max(ylim[1], data.max())]
            # ax.plot(data, label=corrected, color=colors[corrected], linestyle=linestyles[corrected])
            # a bar plot instead
            # ax.bar(np.arange(len(data)), data, color=colors[corrected], alpha=1.0)
            # bars for low, mid, high
            # ax.bar(np.arange(len(low)), data[:len(low)], color=colors['low'], alpha=1.0)
            # ax.bar(np.arange(len(low), len(low) + len(mid)), data[len(low):len(low) + len(mid)], color=colors['mid'], alpha=1.0)
            # ax.bar(np.arange(len(low) + len(mid), len(low) + len(mid) + len(high)), data[len(low) + len(mid):], color=colors['high'], alpha=1.0)
            # simply plot all the bars with the same color
            ax.bar(np.arange(len(data)), data, color=colors['low'], alpha=1.0, align='edge')
            # also add gray shade for mid
        figures += [(fig, ax, deployed, corrected)]

for fig, ax, deployed, corrected in figures:
    # draw a gray background between low and high
    # ax.fill_between([args.low, args.high], ylim[0], ylim[1], color='gray', alpha=0.3)
    # ax.set_xlim(-1, 224)
    # ax.fill_between([len(low)-.5, len(low) + len(mid)-.5], ylim[0], 1.1*ylim[1], color='gray', alpha=0.2, zorder=-1)
    a_low = int(3**(1/p))
    a_high = 190**(1/p)
    ax.fill_between([a_low, a_high], ylim[0], 1.1*ylim[1], color='gray', alpha=0.2, zorder=-1)
    ax.set_xlim(-.2, len(freq_ranges))
    # ax.set_ylim(ylim)
    ax.set_ylim(0, ylim[1] * 1.1)
    # ax.set_yscale('log')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    fig.savefig(args.output_directory + f'/{args.category}_{deployed}_{corrected}.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)

# plt.xlim(0, 64)
# plt.ylim(ylim)

# # plt.yscale('log')
# plt.tight_layout()
# plt.xticks([])
# plt.yticks([])
# # for some reason i still see some (logarithmic) y-ticks
# plt.gca().yaxis.set_major_formatter(plt.NullFormatter())
# plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_minor_locator(plt.NullLocator())

# plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
