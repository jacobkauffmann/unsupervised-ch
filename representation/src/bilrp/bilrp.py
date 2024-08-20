import torch
from tqdm import tqdm
import numpy as np
from zennit.attribution import Gradient
from bilrp.plotting import plot_relevances, clip, get_alpha


def compute_branch(x, model, composite, device='cuda'):
    e = model.forward(x)
    y = e.squeeze()
    n_features = y.shape

    R = []
    for k, yk in tqdm(enumerate(y)):
        z = np.zeros((n_features[0]))
        z[k] = y[k].detach().cpu().numpy().squeeze()
        r_proj = (
            torch.FloatTensor((z.reshape([1, n_features[0], 1, 1])))
            .to(device)
            .data.squeeze(2)
            .squeeze(2)
        )
        model.zero_grad()
        x.grad = None
        with Gradient(model=model, composite=composite) as attributor:
            out, relevance = attributor(x, r_proj)
        relevance = relevance.squeeze().detach().cpu().numpy()
        R.append(relevance)
        del out, relevance
    return R, e


def pool(X, stride):
    K = [
        torch.nn.functional.avg_pool2d(
            torch.from_numpy(o).unsqueeze(0).unsqueeze(1),
            kernel_size=stride,
            stride=stride,
            padding=0,
        )
        .squeeze()
        .numpy()
        for o in X
    ]
    return K


def compute_rel(r1, r2, poolsize=[8]):
    R = [np.array(r).sum(1) for r in [r1, r2]]
    R = np.tensordot(pool(R[0], poolsize), pool(R[1], poolsize), axes=(0, 0))
    return R


def plot_bilrp(x1, x2, R1, R2, fname=None, normalization_factor='individual'):
    clip_func = lambda x: get_alpha(clip(x, clim1=[-2, 2], clim2=[-20, 20], normalization_factor=normalization_factor),
                                    p=2)
    poolsize = [8]
    R = compute_rel(R1, R2)
    indices = np.indices(R.shape)
    inds_all = [(i, R[i[0], i[1], i[2], i[3]]) for i in indices.reshape((4, np.prod(indices.shape[1:]))).T]
    plot_relevances(inds_all, x1, x2, clip_func, poolsize, curvefac=2.5, fname=fname)


def projection_conv(input_dim, embedding_size=2048):
    pca = torch.nn.Sequential(
        *[torch.nn.Flatten(), torch.nn.Conv2d(input_dim, embedding_size, (1, 1), bias=False), ])
    return pca
