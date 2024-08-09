# root = '/Volumes/RAM Disk/mvtec_anomaly_detection/'
# identify if this is my local mac or the cluster
import os
if os.path.exists('/Users/jack/data/mvtec_anomaly_detection/'):
    root = '/Users/jack/data/mvtec_anomaly_detection/'
else:
    root = '/data/mvtec_anomaly_detection/'

from src.mvtec import load_data
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch

import yaml
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
config = load_config('workflow/config/config.yaml')
imsize = config['imsize']

CATEGORIES = ['bottle', 'capsule', 'pill', 'toothbrush', 'wood']
# CATEGORIES = ['toothbrush']
ALL_CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper']
normalize = Normalize(mean=.5, std=.5)
unnormalize = Normalize(mean=-1, std=2)
transform = Compose([
    Resize(imsize),
    ToTensor(),
    normalize
])
target_transform = Compose([
    Resize(imsize),
    ToTensor()
])

def load_data_tensor(category, *args, **kwargs):
    trainset, testset = load_data(the_class=category, root=root, *args, **kwargs)

    Xtrain = torch.stack([x for x in trainset])

    Xtest, ytest, Seg = [], [], []
    for x, seg in testset:
        Xtest.append(x)
        ytest.append(seg.sum() > 0)
        Seg.append(seg)
    Xtest = torch.stack(Xtest)
    ytest = torch.tensor(ytest)
    Seg = torch.stack(Seg)

    return Xtrain, Xtest, ytest, Seg
