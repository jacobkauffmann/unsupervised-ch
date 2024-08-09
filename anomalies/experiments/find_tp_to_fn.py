from src.models import models
from src.data import load_data_tensor, CATEGORIES, transform, target_transform
from src.artifacts import artifact_transform
from src.utils import split_for_supervised, immono, data2img
from src.metrics import instance_metric, optimal_threshold, false_negative_rate
from src.correction import power_spectrum, normalize_spectrum, ApplyPerChannel, V
from src.correction import correction_layer

import torch as tr
device = tr.device('cpu')

import sys
import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.metrics import confusion_matrix

artifact = 'cv2_resize'

train_transform = artifact_transform(artifact)
for category in CATEGORIES:
    print(category)
    Xtrain, Xtest, ytest, Seg = load_data_tensor(category, transform=train_transform, target_transform=target_transform)

    deployed_transform = artifact_transform('none')
    _, Xtest_deployed, _, _ = load_data_tensor(category, transform=deployed_transform, target_transform=target_transform)

    Xtrain, Xtest, Xtest_deployed = Xtrain.to(device), Xtest.to(device), Xtest_deployed.to(device)

    model = models['D2Neighbors']()

    model.fit(Xtrain)

    ypred_evaluation = model(Xtest)
    ypred_deployed = model(Xtest_deployed)

    threshold = optimal_threshold(ytest, ypred_evaluation)
    ypred_evaluation = ypred_evaluation > threshold
    ypred_deployed = ypred_deployed > threshold

    # find samples (indices) that are true positives in evaluation but false negatives in deployed
    tp = ypred_evaluation & ytest
    fn = ~ypred_deployed & ytest

    idc = tr.nonzero(tp & fn).flatten()

    print(f'Found {len(idc)} samples that are true positives in evaluation but false negatives in deployed')
    print(f'Indices: {idc}')

    # also save these images
    for idx in idc:
        immono(data2img(Xtest[idx]).numpy(), filename=f'results/tp_to_fn/{category}_{idx}_evaluation.png')
        immono(data2img(Xtest_deployed[idx]).numpy(), filename=f'results/tp_to_fn/{category}_{idx}_deployed.png')
