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

gamma = 'scale'
C = 1e-9
kernel = 'linear'

for category in CATEGORIES:
    print(category)
    scores[category] = {}
    Xtrain, Xtest, ytest, Seg = load_data_tensor(category, transform=transform, target_transform=transform)
    Xtrain_corrected = correction(Xtrain)

    # supervised
    # model = KernelClustering()
    model = Supervised(kernel=kernel, C=C, gamma=gamma)
    XXtrain, yytrain, XXtest, yytest = model.train_test_split(Xtrain, Xtest, ytest)
    XXtrain_corrected = correction(XXtrain)
    # svs, ysvs = model.fit_svs(XXtrain.flatten(1), yytrain)
    # y_score = np.array([model.fc(x.unsqueeze(0), 1, svs, ysvs) for x in XXtest])
    model.fit(XXtrain, yytrain)
    y_score = model.predict(XXtest)
    threshold = optimal_threshold(yytest.numpy(), y_score, beta=1)
    scores[category]['original'] = instance_metric(yytest.numpy(), y_score > threshold)
    print(f'  supervised: %.2f'%(100*scores[category]['original']))

    mean = XXtrain.mean(0)
    common_args['ref'] = mean

    for artifact_name, artifact in artifacts.items():
        # model_corrected = KernelClustering()
        model_corrected = Supervised(kernel=kernel, C=C, gamma=gamma)
        # svs_corrected, ysvs_corrected = model_corrected.fit_svs(XXtrain_corrected.flatten(1), yytrain)
        model_corrected.fit(XXtrain_corrected, yytrain)
        y_score, y_score_corrected = [], []
        for i, x in enumerate(XXtest):
            x_ = artifact(x, **common_args)
            # ypred.append(model.fc(x_.unsqueeze(0), 1, svs, ysvs) > threshold)
            y_score.append(model.predict(x_.unsqueeze(0)))
            # y_score_corrected.append(model_corrected.fc(correction(x_.unsqueeze(0)), 1, svs_corrected, ysvs_corrected))
            y_score_corrected.append(model_corrected.predict(correction(x_.unsqueeze(0))))
        y_score = np.array(y_score)
        y_score_corrected = np.array(y_score_corrected)
        ypred = np.array(y_score > threshold)
        optimal_threshold_corrected = optimal_threshold(yytest.numpy(), model_corrected.predict(correction(XXtest)))
        ypred_corrected = np.array(y_score_corrected > optimal_threshold_corrected)
        scores[category][f'{artifact_name}'] = instance_metric(yytest.numpy(), ypred)
        scores[category][f'{artifact_name}_corrected'] = instance_metric(yytest.numpy(), ypred_corrected)
        print(f'  {artifact_name} supervised: %.2f'%(100*scores[category][f'{artifact_name}']))
        print(f'  {artifact_name} supervised corrected: %.2f'%(100*scores[category][f'{artifact_name}_corrected']))

        # plot the distribution over predictions as a histogram, colored by ground truth class
        import matplotlib.pyplot as plt
        for y_, title in zip([y_score - threshold, y_score_corrected - optimal_threshold_corrected], ['artifact', 'corrected']):
            min_val = min(np.min(y_[yytest == 0]), np.min(y_[yytest == 1]))
            max_val = max(np.max(y_[yytest == 0]), np.max(y_[yytest == 1]))

            bins = np.linspace(min_val, max_val, num=20)  # 20 Bins

            plt.figure()
            plt.hist(y_[yytest == 0], bins=bins, alpha=.5, color='r', label='normal')
            plt.hist(y_[yytest == 1], bins=bins, alpha=.5, color='b', label='anomaly')
            plt.legend()
            plt.savefig(f'{path}/supervised_{category}_{artifact_name}_{title}_histogram.png')
            plt.close()

with open(f'{path}/scores_supervised.json', 'w') as f:
    json.dump(scores, f)
