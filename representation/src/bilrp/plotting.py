from matplotlib import pyplot as plt
import numpy as np
import numpy

NORMALIZATION_FACTORS = {
    'covid': 0.1423813963010631,
}


def clip(R, clim1, clim2, normalization_factor='individual'):
    delta = list(np.array(clim2) - np.array(clim1))

    if normalization_factor == 'individual':
        Rnorm = np.mean(R ** 4) ** 0.25
    else:
        if normalization_factor in NORMALIZATION_FACTORS:
            Rnorm = NORMALIZATION_FACTORS[normalization_factor]
            Rnorm = np.sqrt(np.mean(R ** 4) ** 0.25) * np.sqrt(Rnorm)
        else:
            raise ValueError('unknown normalization factor')

    R = R / Rnorm  # normalization
    R = R - np.clip(R, clim1[0], clim1[1])  # sparsification
    R = np.clip(R, delta[0], delta[1]) / delta[1]  # thresholding
    return R


def get_alpha(x, p=1):
    x = x ** p
    return x


def plot_relevances(c, x1, x2, clip_func, stride, fname=None, curvefac=1.):
    h, w, channels = x1.shape if len(x1.shape) == 3 else list(x1.shape) + [1]
    wgap, hpad = int(0.05 * w), int(0.6 * w)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.ylim(-hpad - 2, h + hpad + 1)
    plt.xlim(0, (w + 2) * 2 + wgap + 1)

    x1 = x1.reshape(h, w, channels).squeeze()
    x2 = x2.reshape(h, w, channels).squeeze()

    border_w = np.zeros((1, w, 4))
    border_h = np.zeros((h + 2, 1, 4))
    border_h[:, :, -1] = 1
    border_w[:, :, -1] = 1

    x1 = np.concatenate([border_h, np.concatenate([border_w, x1, border_w], axis=0), border_h], axis=1)
    x2 = np.concatenate([border_h, np.concatenate([border_w, x2, border_w], axis=0), border_h], axis=1)

    mid = numpy.ones([h + 2, wgap, channels]).squeeze()
    X = numpy.concatenate([x1, mid, x2], axis=1)[
        ::-1]
    plt.imshow(X, cmap='gray', vmin=-1, vmax=1)

    if len(stride) == 2:
        stridex = stride[0]
        stridey = stride[1]
    else:
        stridex = stridey = stride[0]

    relevance_array = np.array([i[1] for i in c])
    indices = [i[0] for i in c]

    alphas = clip_func(relevance_array)
    inds_plotted = []

    for indx, alpha, s in zip(indices, alphas, relevance_array):
        i, j, k, l = indx[0], indx[1], indx[2], indx[3]

        if alpha > 0.:
            xm = int(w / 2) + 6
            xa = stridey * j + (stridey / 2 - 0.5) - xm
            xb = stridey * l + (stridey / 2 - 0.5) - xm + w + wgap
            ya = h - (stridex * i + (stridex / 2 - 0.5))
            yb = h - (stridex * k + (stridex / 2 - 0.5))
            ym = (0.8 * (ya + yb) - curvefac * int(h / 6))
            ya -= ym
            yb -= ym
            lin = numpy.linspace(0, 1, 25)
            plt.plot(xa * lin + xb * (1 - lin) + xm, ya * lin ** 2 + yb * (1 - lin) ** 2 + ym,
                     color='red' if s > 0 else 'blue', alpha=alpha)

        inds_plotted.append(((i, j, k, l), s))

    plt.axis('off')

    if fname:
        plt.tight_layout()
        plt.savefig(fname, dpi=300, transparent=True)
    else:
        plt.show()
    plt.close()
