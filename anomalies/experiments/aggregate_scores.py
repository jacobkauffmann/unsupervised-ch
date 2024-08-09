from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--output', type=str, default='results/scores/scores.json')
args = parser.parse_args()

import json
from src.data import ALL_CATEGORIES

models = ['D2Neighbors', 'PatchCore', 'D2NeighborsL1']
categories = ALL_CATEGORIES
artifacts = ['none']# ['cv2resize', 'gaussiannoise']

the_dict = {}

for model in models:
    the_dict[model] = {}
    for artifact in artifacts:
        the_dict[model][artifact] = {}
        for category in categories:
            with open(f'results/scores/{model}/{artifact}/{category}.json') as f:
                score = json.load(f)
            the_dict[model][artifact][category] = score

with open(args.output, 'w') as f:
    json.dump(the_dict, f)
