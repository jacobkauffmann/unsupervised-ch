import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
from wand.image import Image as WandImage

from src.data import normalize

import yaml
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
config = load_config('workflow/config/config.yaml')
imsize = config['imsize']

def resize_cv2(x, size=(64, 64), interpolation=cv2.INTER_NEAREST, **kwargs):
    # x is PIL image
    open_cv_image = np.array(x)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    open_cv_image = cv2.resize(open_cv_image, size, interpolation=interpolation)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    # turn back to PIL image
    pil_image = Image.fromarray(open_cv_image)
    return pil_image

# transform_cv2 = transforms.Compose([
#     transforms.Lambda(lambda x: resize_cv2(x, size=(64, 64), interpolation=cv2.INTER_CUBIC)),
#     transforms.ToTensor()
# ])

def make_mask(n):
    board = np.zeros((8, 8), dtype=bool)
    if n > 0:
        board[np.random.randint(8), np.random.randint(8)] = True
    for _ in range(1, n):
        candidates = [(y, x) for y in range(8) for x in range(8) if not board[y, x] and
                      ((y > 0 and board[y - 1, x]) or (y < 7 and board[y + 1, x]) or
                       (x > 0 and board[y, x - 1]) or (x < 7 and board[y, x + 1]))]
        if not candidates:
            break
        y, x = candidates[np.random.randint(len(candidates))]
        board[y, x] = True
    return np.kron(board, np.ones((8, 8), dtype=bool))

def checkerboard(x, n, ref, **kwargs):
    mask = torch.from_numpy(make_mask(n)).unsqueeze(0).repeat(3,1,1)
    x_ = torch.where(mask, ref, x).clone()
    return x_

def blur(x, kernel_size=(3,3), sigma=1.0, **kwargs):
    x = x.unsqueeze(0)
    x = torchvision.transforms.functional.gaussian_blur(x, kernel_size, sigma)
    return x.squeeze(0)

def sharpen(x, **kwargs):
    kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    x = x.unsqueeze(0)
    x = torch.nn.functional.conv2d(x, kernel, padding=1, groups=3)
    return x.squeeze(0)

def saltpepper(x, p=0.3, **kwargs):
    min_val = x.min()
    max_val = x.max()
    mask = torch.rand_like(x[0]) < p
    salt = mask & (torch.rand_like(x[0]) < 0.5)
    pepper = mask & ~salt
    x = torch.where(salt.unsqueeze(0), max_val, x)
    x = torch.where(pepper.unsqueeze(0), min_val, x)
    return x

def gaussian_noise(x, std=0.1, **kwargs):
    noise = torch.randn_like(x) * std
    return x + noise

def dither_quantize(pil_image, **kwargs):
    with WandImage.from_array(np.array(pil_image)) as wand_image:
        # wand_image.quantize(number_colors=16, dither='floyd_steinberg')
        with WandImage(width=216, height=144, pseudo='netscape:') as palette:
            wand_image.remap(affinity=palette, method='floyd_steinberg')
        image = np.array(wand_image)
    return Image.fromarray(image)

def quantize(pil_image, **kwargs):
    with WandImage.from_array(np.array(pil_image)) as wand_image:
        # wand_image.quantize(number_colors=16, dither='no')
        with WandImage(width=216, height=144, pseudo='netscape:') as palette:
            wand_image.remap(affinity=palette, method='no')
        image = np.array(wand_image)
    return Image.fromarray(image)

artifacts = {
    'blurring': blur,
    'sharpening': sharpen,
    # 'checkerboard': checkerboard,
    'salt&pepper': saltpepper,
    'gaussiannoise': gaussian_noise,
    'cv2resize': resize_cv2,
    'dither': dither_quantize,
    'quantize': quantize,
    'none': lambda x, **kwargs: x
}



def artifact_transform(artifact, **kwargs):
    if artifact == 'cv2resize':
        return transforms.Compose([
            transforms.Lambda(lambda x: resize_cv2(x, size=(imsize, imsize), interpolation=cv2.INTER_CUBIC)),
            transforms.ToTensor(),
            normalize
        ])

    if artifact == 'dither' or artifact == 'quantize':
        return transforms.Compose([
            transforms.Resize(imsize, imsize),
            transforms.Lambda(lambda x: artifacts[artifact](x, **kwargs)),
            transforms.ToTensor(),
            normalize
        ])

    return transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: artifacts[artifact](x, **kwargs)),
        normalize
    ])
