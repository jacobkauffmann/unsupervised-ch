import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from data import TRUCKS, FISH
from utils.probing import load_embeddings_labels, load_class_names, train_classifier
import os
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--output-dir')
parser.add_argument('--dataset', choices=['trucks', 'fish'])
args = parser.parse_args()

model_name = args.model

results = {}

if args.dataset == 'trucks':
    class_names = load_class_names(TRUCKS)
elif args.dataset == 'fish':
    class_names = load_class_names(FISH)
else:
    raise ValueError()

base = f'resources/features/{args.dataset}'

# train classifier
X_train, y_train = load_embeddings_labels(os.path.join(base, 'std/train',
                                                       model_name + '.npz'))
clf = train_classifier(X_train, y_train, reg=1.0)

os.makedirs(args.output_dir, exist_ok=True)
coefs = {model_name: clf.coef_}
np.savez(os.path.join(args.output_dir, f'{model_name}_probe_weights.npz'), **coefs)

# inference on clean test set
X_test, y_test = load_embeddings_labels(os.path.join(base, 'std/test',
                                                     model_name + '.npz'))
print(X_test.shape)
predictions = clf.predict(X_test)

results[model_name] = {
    'accuracy': accuracy_score(y_test, predictions),
    'predictions': predictions.tolist(),
    'labels': y_test.tolist(),
    'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
    'class_names': class_names
}

if args.dataset == 'trucks':
    # inference on watermark samples
    X_watermark, y_watermark = load_embeddings_labels(os.path.join(base, 'watermark_all/test',
                                                                   model_name + '.npz'))
    watermark_predictions = clf.predict(X_watermark)
    watermark_results = {'watermark_accuracy': accuracy_score(y_watermark, watermark_predictions),
                         'watermark_confusion_matrix': confusion_matrix(y_watermark, watermark_predictions).tolist(),
                         'watermark_predictions': watermark_predictions.tolist(),
                         'watermark_labels': y_watermark.tolist()}
    results[model_name] = {**results[model_name], **watermark_results}

with open(os.path.join(args.output_dir, f'{model_name}.json'), 'w') as f:
    json.dump(results, f)
