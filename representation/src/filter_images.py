import sys
sys.path.append('.')
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torch
from data import ImagenetSubset, BONE_FISH, TRUCKS, FISH
import argparse
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='resources/imagenette2-320')
parser.add_argument('--output', default='resources/person_indicator/bone-fish-50-50')
parser.add_argument('--dataset', default='imagenette')
parser.add_argument('--split', default='val')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

name = args.dataset
val_path = os.path.join(args.data_root, args.split)

if name == 'imagenette':
    dataset = ImageFolder(root=val_path)
elif name == 'bone-fish':
    dataset = ImagenetSubset(root=val_path, classes=BONE_FISH)
elif name == 'trucks':
    dataset = ImagenetSubset(root=val_path, classes=TRUCKS)
elif name == 'fish':
    dataset = ImagenetSubset(root=val_path, classes=FISH)
else:
    raise ValueError()

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.to(args.device)
model.eval()

person_indicator = torch.zeros(len(dataset), dtype=torch.bool)
for idx in tqdm(range(len(dataset))):
    img, label = dataset[idx]
    img = torch.tensor(np.array(img))
    img = torch.permute(img, (2, 0, 1))

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img).to(args.device)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    is_person = 'person' in labels

    if is_person:
        person_indicator[idx] = True

np.save(os.path.join(args.output), person_indicator.numpy())
