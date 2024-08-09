from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--output', type=str, default='results/scores/cm.json')
args = parser.parse_args()

import json
from src.data import CATEGORIES
from src.metrics import instance_metric, compute_sample_weight

models = ['D2Neighbors', 'PatchCore', 'D2NeighborsL1', 'D2NeighborsL4']
categories = CATEGORIES
artifacts = ['cv2resize']#, 'gaussiannoise']

the_dict = {}

for model in models:
    the_dict[model] = {}
    for artifact in artifacts:
        the_dict[model][artifact] = {}
        for category in categories:
            with open(f'results/scores/{model}/{artifact}/{category}.json') as f:
                the_json = json.load(f)
            cm_evaluation = the_json['cm_evaluation']
            cm_deployed = the_json['cm_deployed']
            cm_corrected = the_json['cm_corrected']
            cm_corrected_evaluation = the_json['cm_corrected_evaluation']
            the_dict[model][artifact][category] = {
                'cm_evaluation': cm_evaluation,
                'cm_deployed': cm_deployed,
                'cm_corrected': cm_corrected,
                'cm_corrected_evaluation': cm_corrected_evaluation
            }

with open(args.output, 'w') as f:
    json.dump(the_dict, f)

def extract_confusion_matrix(cm_dict):
    return {
        category: cm_dict[category]["cm_deployed"]
        for category in cm_dict
    }, {
        category: cm_dict[category]["cm_corrected"]
        for category in cm_dict
    }, {
        category: cm_dict[category]["cm_evaluation"]
        for category in cm_dict
    }, {
        category: cm_dict[category]["cm_corrected_evaluation"]
        for category in cm_dict
    }

def calculate_average_f1(cm_dict):
    f1_scores = []
    for category in cm_dict:
        tn, fp = cm_dict[category][0]
        fn, tp = cm_dict[category][1]
        y_true = [0] * (tn + fp) + [1] * (fn + tp)
        y_pred = [0] * tn + [1] * fp + [0] * fn + [1] * tp
        f1 = instance_metric(y_true, y_pred, beta=1)
        print(f"    {category:<10}: {f1.__round__(2)}")
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)

# Extract and calculate the required information
results = []
for model in the_dict:
    print(model)
    if "cv2resize" in the_dict[model]:
        cm_deployed, cm_corrected, cm_evaluation, cm_corrected_evaluation = extract_confusion_matrix(the_dict[model]["cv2resize"])
        print('  deployed')
        avg_f1_deployed = calculate_average_f1(cm_deployed)
        print(f'    {"mean":<10}: {avg_f1_deployed.__round__(2)}')
        print('  corrected')
        avg_f1_corrected = calculate_average_f1(cm_corrected)
        print(f'    {"mean":<10}: {avg_f1_corrected.__round__(2)}')
        print('  evaluation')
        avg_f1_evaluation = calculate_average_f1(cm_evaluation)
        print(f'    {"mean":<10}: {avg_f1_evaluation.__round__(2)}')
        print('  corrected evaluation')
        avg_f1_corrected_evaluation = calculate_average_f1(cm_corrected_evaluation)
        print(f'    {"mean":<10}: {avg_f1_corrected_evaluation.__round__(2)}')
        # results.append({
        #     "Model": model,
        #     "Evaluation": avg_f1_evaluation,
        #     "Corrected Evaluation": avg_f1_corrected_evaluation,
        #     "Deployed": avg_f1_deployed,
        #     "Corrected": avg_f1_corrected,
        # })

# Create DataFrame and display the result
# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# df = pd.DataFrame(results)
# print(results)
