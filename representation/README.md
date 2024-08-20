# Representation Learning Experiments

This folder contains the code to reproduce our results for the representation learning models.
Models that are use in the paper are: `r50-sup`, `r50-barlowtwins`, `r50-clip`, `simclr-rn50`.

### Install dependencies

```bash
pip install  -r requirements.txt
```

### Compute embeddings

Embeddings for the different ResNet-50 models can be extracted with the following script.

```bash
python extract_embeddings.py --data-root <path_to_imagenet_root> \
            --model <model_name> \
            --output-dir <output_dir> \
            --dataset <trucks|fish> \
            --device cuda \
            --split <train|test>

```

### Generate BiLRP Heatmaps

BILRP heatmaps can be generated with first computing LRP relevances (`compute_bilrp.py`) and
then plotting the result (`plots/plot_bilrp.py`).

### Linear Classifiers

To train linear classifiers on the extracted embeddings, `linear_probing.py` can be used. This generates
json files with the predictions of the linear classifiers.

### Plot classifier results

With the notebook `plots/representation.ipynb`, the linear probing results can then be analyzed and plotted.

### T-SNE plots
The T-SNE plots can be generated from the extracted features with `plots/fish_tsne.py` and `plots/trucks_tsne.py`.

