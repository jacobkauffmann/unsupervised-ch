from sklearn.linear_model import LogisticRegression
import json
import numpy as np

Array = np.ndarray


def train_classifier(
        train_features: Array,
        train_labels: Array,
        reg: float,
):
    clf = LogisticRegression(random_state=1, max_iter=1000, fit_intercept=False,
                             C=reg, class_weight='balanced').fit(train_features, train_labels)
    return clf


def load_embeddings_labels(path):
    arr = np.load(path)
    X = arr['embeddings']
    y = arr['labels']
    return X, y


def load_class_names(imagenet_classes):
    with open('resources/imagenet_class_index.json') as f:
        class_index = json.load(f)
    class_dict = {item[0]: item[1] for item in class_index.values()}
    classes = [class_dict[wnid] for wnid in imagenet_classes]
    return classes
