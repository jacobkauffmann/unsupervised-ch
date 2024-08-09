import json

# Load the data
with open('results/scores/confusion_matrices.json', 'r') as f:
    data = json.load(f)

# Function to calculate F1 score
def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1

# Function to calculate metrics from confusion matrix
def calculate_metrics(cm):
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]
    tn = cm[0][0]
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    return tp, fn, tn, fp, fnr

# Initialize results containers
f1_scores = {}
average_metrics = {}

# Iterate through the data to calculate F1 scores and metrics for cv2resize
for algorithm, transformations in data.items():
    if "cv2resize" in transformations:
        f1_scores[algorithm] = {}
        avg_metrics_eval = {'TP': 0, 'FN': 0, 'TN': 0, 'FP': 0, 'FNR': 0, 'count': 0}
        avg_metrics_deploy = {'TP': 0, 'FN': 0, 'TN': 0, 'FP': 0, 'FNR': 0, 'count': 0}
        avg_metrics_corrected = {'TP': 0, 'FN': 0, 'TN': 0, 'FP': 0, 'FNR': 0, 'count': 0}

        for category, matrices in transformations["cv2resize"].items():
            f1_scores[algorithm][category] = {}
            for matrix_type, cm in matrices.items():
                tp, fn, tn, fp = cm[1][1], cm[1][0], cm[0][0], cm[0][1]
                f1 = calculate_f1(tp, fp, fn)

                if matrix_type == 'cm_evaluation':
                    avg_metrics_eval['TP'] += tp
                    avg_metrics_eval['FN'] += fn
                    avg_metrics_eval['TN'] += tn
                    avg_metrics_eval['FP'] += fp
                    avg_metrics_eval['FNR'] += calculate_metrics(cm)[4]
                    avg_metrics_eval['count'] += 1
                    f1_scores[algorithm][category]['eval'] = f1

                elif matrix_type == 'cm_deployed':
                    avg_metrics_deploy['TP'] += tp
                    avg_metrics_deploy['FN'] += fn
                    avg_metrics_deploy['TN'] += tn
                    avg_metrics_deploy['FP'] += fp
                    avg_metrics_deploy['FNR'] += calculate_metrics(cm)[4]
                    avg_metrics_deploy['count'] += 1
                    f1_scores[algorithm][category]['deploy'] = f1

                elif matrix_type == 'cm_corrected':
                    avg_metrics_corrected['TP'] += tp
                    avg_metrics_corrected['FN'] += fn
                    avg_metrics_corrected['TN'] += tn
                    avg_metrics_corrected['FP'] += fp
                    avg_metrics_corrected['FNR'] += calculate_metrics(cm)[4]
                    avg_metrics_corrected['count'] += 1
                    f1_scores[algorithm][category]['corrected'] = f1

        average_metrics[algorithm] = {
            'TP': f"{avg_metrics_eval['TP'] / avg_metrics_eval['count']:.2f} / {avg_metrics_deploy['TP'] / avg_metrics_deploy['count']:.2f} / {avg_metrics_corrected['TP'] / avg_metrics_corrected['count']:.2f}",
            'FN': f"{avg_metrics_eval['FN'] / avg_metrics_eval['count']:.2f} / {avg_metrics_deploy['FN'] / avg_metrics_deploy['count']:.2f} / {avg_metrics_corrected['FN'] / avg_metrics_corrected['count']:.2f}",
            'TN': f"{avg_metrics_eval['TN'] / avg_metrics_eval['count']:.2f} / {avg_metrics_deploy['TN'] / avg_metrics_deploy['count']:.2f} / {avg_metrics_corrected['TN'] / avg_metrics_corrected['count']:.2f}",
            'FP': f"{avg_metrics_eval['FP'] / avg_metrics_eval['count']:.2f} / {avg_metrics_deploy['FP'] / avg_metrics_deploy['count']:.2f} / {avg_metrics_corrected['FP']:.2f}",
            'FNR': f"{avg_metrics_eval['FNR'] / avg_metrics_eval['count']:.2f} / {avg_metrics_deploy['FNR'] / avg_metrics_deploy['count']:.2f} / {avg_metrics_corrected['FNR'] / avg_metrics_corrected['count']:.2f}"
        }

# LaTeX table for F1 scores
latex_f1 = "\\begin{table}[h]\n"
latex_f1 += "    \\centering\n"
latex_f1 += "    \\begin{tabular}{l|llll}\n"
latex_f1 += "        \\textbf{Category} & \\textbf{D2Neighbors} & \\textbf{D2NeighborsL1} & \\textbf{D2NeighborsL4} & \\textbf{PatchCore} \\\\ \\hline\n"

categories = set(category for alg in f1_scores for category in f1_scores[alg])
algorithms = ['D2Neighbors', 'D2NeighborsL1', 'D2NeighborsL4', 'PatchCore']

for category in categories:
    latex_f1 += f"        {category} "
    for alg in algorithms:
        if alg in f1_scores and category in f1_scores[alg]:
            scores = f1_scores[alg][category]
            latex_f1 += f" & {scores.get('eval', 'N/A'):.2f} / {scores.get('deploy', 'N/A'):.2f} / {scores.get('corrected', 'N/A'):.2f}"
        else:
            latex_f1 += " & N/A / N/A / N/A"
    latex_f1 += " \\\\\n"

latex_f1 += "    \\end{tabular}\n"
latex_f1 += "    \\caption{Combined F1 Scores by Algorithm and Category for cv2resize. The first number in each cell represents the results at evaluation time, the second number at deployment time (with improved data quality), and the third number represents our blurring based robustified version.}\n"
latex_f1 += "    \\label{tab:f1_scores_cv2resize}\n"
latex_f1 += "\\end{table}\n"

# LaTeX table for average metrics (transposed)
latex_avg = "\\begin{table}[h]\n"
latex_avg += "    \\centering\n"
latex_avg += "    \\begin{tabular}{l|l|l|l|l}\n"
latex_avg += "        \\textbf{Metric} & \\textbf{D2Neighbors} & \\textbf{D2NeighborsL1} & \\textbf{D2NeighborsL4} & \\textbf{PatchCore} \\\\ \\hline\n"

metrics = ['TP', 'FN', 'TN', 'FP', 'FNR']

for metric in metrics:
    latex_avg += f"        {metric} "
    for alg in algorithms:
        if alg in average_metrics:
            latex_avg += f" & {average_metrics[alg][metric]}"
        else:
            latex_avg += " & N/A"
    latex_avg += " \\\\\n"

latex_avg += "    \\end{tabular}\n"
latex_avg += "    \\caption{Average Metrics by Algorithm for cv2resize. The first number in each cell represents the results at evaluation time, the second number at deployment time (with improved data quality), and the third number represents our blurring based robustified version.}\n"
latex_avg += "    \\label{tab:avg_metrics_cv2resize}\n"
latex_avg += "\\end{table}\n"

# Output LaTeX code
print(latex_f1)
print(latex_avg)
