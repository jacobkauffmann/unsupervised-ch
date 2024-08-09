from src.models import MahalanobisKDE, Supervised
from src.data import load_data_tensor, CATEGORIES, transform
from src.metrics import optimal_threshold, instance_metric
from src.utils import immono, data2img, anomaly_boundary
from src.artifacts import artifacts
from src.correction import Correction_circ as Correction
import torch as tr
import numpy as np
import os
import json

from src.utils import immono, data2img

correction = Correction(low=2, high=20)

scores = {}
path = f'results/scores_with_artifacts'
if not os.path.exists(path):
    os.makedirs(path)

common_args = {
    'n': 32,
    'ref': None,
    'kernel_size': (31, 31),
    'sigma': 1.0,
}

for category in CATEGORIES:
    print(category)
    scores[category] = {}
    Xtrain, Xtest, ytest, Seg = load_data_tensor(category, transform=transform, target_transform=transform)
    Xtrain_corrected = correction(Xtrain)

    # D2Neighbors
    model = MahalanobisKDE()
    model.fit(Xtrain)
    y_score = np.array([model(x).item() for x in Xtest])
    threshold = optimal_threshold(ytest.numpy(), y_score, beta=1)
    scores[category]['original'] = instance_metric(ytest.numpy(), y_score > threshold)
    print(f'  original: %.2f'%(100*scores[category]['original']))

    mean = Xtrain.mean(0)
    common_args['ref'] = mean

    for artifact_name, artifact in artifacts.items():
        model_corrected = MahalanobisKDE()
        model_corrected.fit(Xtrain_corrected)
        ypred, y_score_corrected = [], []
        for i, x in enumerate(Xtest):
            x_ = artifact(x, **common_args)
            if i < 5:
                immono(data2img(x_).numpy(), filename=f'{path}/{category}_{artifact_name}_{i}.png', do_plot=False)
            ypred.append(model(x_).item() > threshold)
            y_score_corrected.append(model_corrected(correction(x_.unsqueeze(0)).squeeze()).item())
        ypred = np.array(ypred)
        optimal_threshold_corrected = optimal_threshold(ytest.numpy(), [model_corrected(correction(x.unsqueeze(0)).squeeze()).item() for x in Xtest])
        ypred_corrected = np.array(y_score_corrected > optimal_threshold_corrected)
        scores[category][artifact_name] = instance_metric(ytest.numpy(), ypred)
        scores[category][f'{artifact_name}_corrected'] = instance_metric(ytest.numpy(), ypred_corrected)
        print(f'  {artifact_name}: %.2f'%(100*scores[category][artifact_name]))
        print(f'  {artifact_name} corrected: %.2f'%(100*scores[category][f'{artifact_name}_corrected']))

with open(f'{path}/scores_d2neighbors.json', 'w') as f:
    json.dump(scores, f)
