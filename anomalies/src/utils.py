import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
import torch as tr

from src.data import unnormalize

C = np.array([[0,0,1],[1,1,1],[1,0,0]])
cm = mpl.colors.LinearSegmentedColormap.from_list('', C)

print_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
print_cmap[:,0:3] *= 0.85
print_cmap = ListedColormap(print_cmap)

def data2img(x):
    if isinstance(x, np.ndarray):
        x = tr.tensor(x)
    if x.shape[0] == 3:
        x = unnormalize(x.unsqueeze(0)).squeeze()
        x = x.clamp(min=0, max=1)
    return x.permute(1,2,0).data.squeeze()

def immono(x, cmap=print_cmap, pn=True, vmin=None, vmax=None, filename=None, do_plot=True):
    if vmin is None:
        vmin = -abs(x).max()
    if vmax is None:
        vmax = abs(x).max()

    if do_plot:
        plt.imshow(x, cmap=cmap, vmin=vmin if pn else None, vmax=vmax if pn else None)
        plt.xticks([]), plt.yticks([])
        plt.axis('off')

    if filename is not None:
        plt.imsave(filename, x, cmap=cmap, vmin=vmin if pn else None, vmax=vmax if pn else None)

def anomaly_boundary(x, seg, border_width=2):
    x = x.clone()
    border = find_boundaries(seg.bool().numpy(), mode='outer')
    border = binary_dilation(border, iterations=border_width)
    border = tr.tensor(border).squeeze()
    x[border] = tr.tensor([1.,.3,.3])
    return x

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        tr.manual_seed(random_state)
    I = tr.randperm(X.shape[0])
    Xtrain = X[I[:-int(test_size*X.shape[0])]]
    Xtest = X[I[-int(test_size*X.shape[0]):]]
    ytrain = y[I[:-int(test_size*X.shape[0])]]
    ytest = y[I[-int(test_size*X.shape[0]):]]
    return Xtrain, Xtest, ytrain, ytest

def split_for_supervised(Xtrain, Xtest, ytest, test_size=0.2, random_state=None):
    if random_state is not None:
        tr.manual_seed(random_state)
    n_train = len(Xtrain)
    I = tr.randperm(len(ytest))
    positives = I[ytest[I] == 1]
    negatives = I[ytest[I] == 0]
    Xtrain = tr.cat([Xtrain, Xtest[positives[:int(test_size*len(positives))]]])
    ytrain = tr.cat([tr.zeros(n_train), tr.ones(int(test_size*len(positives)))])
    Xtest = tr.cat([Xtest[negatives], Xtest[positives[int(test_size*len(positives)):]]])
    ytest = tr.cat([tr.zeros(len(negatives)), tr.ones(len(positives)-int(test_size*len(positives)))])
    return Xtrain, Xtest, ytrain, ytest

def interval_cutting(A, B, f, tol=1e-8, verbose=False, max_steps=1000):
    A_, B_ = A, B
    current = (A + B)/2
    steps = 0
    slack = f(current)
    while (B-A)*abs(slack) > tol and steps < max_steps and A != B:
        if slack > 0:
            A = current
        else:
            B = current
        current = (A + B)/2
        slack = f(current)
        steps = steps + 1
    if verbose:
        print('interval cutting converged after %d steps with slack %.2f at %.2f'%(steps,slack,current))
    if steps == 1000:
        print('interval cutting did not converge after %d steps, slack = %.2f'%(max_steps, slack))
    if current == A_:
        print('interval cutting converged to lower limit')
    if current == B_:
        print('interval cutting converged to upper limit')
    return current
