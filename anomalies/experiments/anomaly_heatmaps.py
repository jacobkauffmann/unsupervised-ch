from src.models import MahalanobisKDE
from src.data import load_data_tensor, CATEGORIES, transform
from src.metrics import optimal_threshold, instance_metric
from src.utils import immono, data2img, anomaly_boundary
import os
import numpy as np
import torch
import json

scores = {}
path = f'results/anomaly_heatmaps'

for category in CATEGORIES:
    print(category)
    scores[category] = {}
    Xtrain, Xtest, ytest, Seg = load_data_tensor(category, transform=transform, target_transform=transform)
    model = MahalanobisKDE()
    model.fit(Xtrain)
    y_score = np.array([model(x).item() for x in Xtest])
    threshold = optimal_threshold(ytest.numpy(), y_score, beta=1)
    ypred = torch.from_numpy(y_score > threshold)

    true_positives = (ypred & ytest).nonzero().squeeze()

    category_path = f'{path}/{category}'
    if not os.path.exists(category_path):
        os.makedirs(category_path)
    for i in true_positives:
        R = model.explain(Xtest[i]).detach().sum(0).numpy()
        img = anomaly_boundary(data2img(Xtest[i]), Seg[i]).numpy()
        # img = data2img(Xtest[i]).numpy()

        immono(img, filename=f'{category_path}/{i}_img.png', do_plot=False)
        immono(R, filename=f'{category_path}/{i}_hm.png', do_plot=False)
