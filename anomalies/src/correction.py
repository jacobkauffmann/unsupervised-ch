import torch as tr
import torch
import torch.nn as nn
from torchvision.transforms import Compose, GaussianBlur, functional as F
import numpy as np

device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
# device = torch.device('cpu')

import yaml
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
config = load_config('workflow/config/config.yaml')
imsize = config['imsize']
dct_basis_path = config['dct_basis_path']

V = tr.load(dct_basis_path).to(device) # shape (imsize, imsize)

def data2freq(x):
    return V @ x @ V.T

def freq2data(z):
    return V.T @ z @ V

# y, x = np.ogrid[:imsize, :imsize]
# radius_int = np.sqrt(x**2 + y**2).astype(int)
# masks = tr.tensor(np.array([(radius_int >= i) & (radius_int < i+1) for i in range(radius_int.max() + 1)])).reshape(-1, imsize, imsize)

# we change to zigzag
def zigzag(n):
    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)
    xs = range(n)
    order = {index: n for n, index in enumerate(sorted(
        ((x, y) for x in xs for y in xs),
        key=compare
    ))}
    # create a matrix that has the "n"s in the right order
    A = np.zeros((n, n), dtype=int)
    for (x, y), n in order.items():
        A[x, y] = n
    return A
zigzag_mat = zigzag(imsize)
zigzag_ind = np.unravel_index(np.argsort(zigzag_mat, axis=None), zigzag_mat.shape)

# def power_spectrum(z):
#     M = masks.reshape(-1, imsize*imsize).float()
#     # M = M.float() / M.sum(1, keepdims=True)
#     return z @ M.T

def power_spectrum(z):
    z = z.reshape(imsize, imsize)[zigzag_ind[0], zigzag_ind[1]]
    return z

def normalize_spectrum(spectra):
    # return  (spectra / spectra.mean(1, keepdims=True)).mean(0)
    # trick to avoid division by zero:
    spectra = spectra.mean(0)
    norm = spectra.norm()
    spectra = spectra / norm
    # spectra = (spectra / norm).mean(0)
    # return spectra.log()
    return spectra
    # return spectra.mean(0)
    # log_sectra = spectra.log()
    # return log_sectra.mean(0)
    # return (log_sectra - log_sectra.mean(1, keepdim=True)).mean(0)

# ApplyPerChannel is not needed, data2freq and freq2data are enough
# class ApplyPerChannel(nn.Module):
#     def __init__(self, V=V):
#         super(ApplyPerChannel, self).__init__()
#         self.V = V

#     def forward(self, x):
#         assert x.dim() == 4, f'x.dim() = {x.dim()}'
#         assert x.shape[2] == x.shape[3], f'x.shape = {x.shape}'
#         return self.V @ x @ self.V.T

# def create_circular_mask(h, w, center=None, low=0, high=90):
#     if center is None:  # use the center of the image
#         center = [0, 0]

#     Y, X = np.ogrid[:h, :w]
#     dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

#     # Create the inner and outer masks
#     inner_mask = (low - dist_from_center).clip(min=0, max=1)
#     outer_mask = (dist_from_center - high).clip(min=0, max=1)

#     # Combine the masks
#     mask = 1 - (inner_mask + outer_mask)

#     return mask


class Correction_circ(tr.nn.Module):
    def __init__(self, low=0, high=90):
        super().__init__()
        self.V = V.clone()
        self.low = low
        self.high = high

    def forward(self, img):
        x = self.V @ img @ self.V.T
        # x = x * (masks[self.low:self.high].sum(0) > 0)
        x = x * (zigzag_mat < self.high) * (zigzag_mat >= self.low)
        x = self.V.T @ x @ self.V
        return x


def correction(transform):
    return Compose([transform, GaussianBlur(kernel_size=(5, 5))])

def correction_layer(other_layer=lambda x: x):
    # as of May 2024, torchvision.transforms.functional.gaussian_blur is implemented as a convolution
    def corr(x):
        x = other_layer(x)
        x = x.reshape(-1, 3, imsize, imsize)
        x = F.gaussian_blur(x, kernel_size=(11, 11))
        # x = x.reshape(-1, 3*64*64)
        return x
    return corr
