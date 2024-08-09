from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--category', type=str, default='toothbrush')
parser.add_argument('--model', type=str, default='D2Neighbors')
parser.add_argument('--output', type=str, default='results/scores/scores.json')
args = parser.parse_args()

from src.models import models
from src.data import load_data_tensor, transform, target_transform
from src.metrics import optimal_threshold, instance_metric

import json

Xtrain, Xtest, ytest, Seg = load_data_tensor(args.category, transform=transform, target_transform=target_transform)

model = models[args.model]()

model.fit(Xtrain)
y_score = model(Xtest)
threshold = optimal_threshold(ytest, y_score)
y_pred = y_score > threshold
score = instance_metric(ytest, y_pred)

with open(args.output, 'w') as f:
    json.dump(score, f)
