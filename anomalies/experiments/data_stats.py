from src.data import load_data_tensor, ALL_CATEGORIES, transform, target_transform

output_string = '\\begin{tabular}{l|ccc}\n'
output_string += 'Category & Train & Test normal & Test anomalous \\\\\n'
output_string += '\hline\n'

for category in ALL_CATEGORIES:
    Xtrain, Xtest, ytest, Seg = load_data_tensor(category, transform=transform, target_transform=target_transform)
    category = category.replace('_', ' ')
    output_string += f'{category} & {Xtrain.shape[0]} & {ytest.eq(0).sum().item()} & {ytest.eq(1).sum().item()} \\\\\n'

output_string += '\end{tabular}'

print(output_string)
