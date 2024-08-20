# THE CLEVER HANS EFFECT IN UNSUPERVISED LEARNING

## Authors:
- Jacob Kauffmann
- Jonas Dippel
- Lukas Ruff
- Wojciech Samek
- Gregoire Montavon
- Klaus-Robert Müller

## Usage:
### 1. Data Preparation

Download the required datasets:
- NIH CXR8: [https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0](https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0)
- GitHub COVID-19 image collection: [https://github.com/ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- ImageNet: [https://www.image-net.org/download.php](https://www.image-net.org/download.php)
- MVTec-AD: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

Place the datasets in the `data/` directory.

### 2. Running the experiments

#### 2.1 Software dependencies
- All experiments are implemented in Python.
- Main dependencies: `torch`, `torchvision`, `ipython`, `matplotlib`, `scikit-learn`, `scipy`, `numpy`, `Pillow`, `opencv-python-headless`.
- Experiments provide individual `requirements.txt` files.

#### 2.2 COVID-19
- Navigate to the `radiology/` directory.
- Open `covid19.ipynb` with `ipython`.
- Run all cells.
- Results can be found in the cell outputs and the `results/` sub-directory.

#### 2.3 Anomaly Detection
- Additional dependency: [Snakemake](https://snakemake.github.io/).
- Navigate to the `anomalies/` directory.
- Run `snakemake --cores 12`.
- Results can be found in the `results/` sub-directory.

#### 2.4 Representation Learning
- Navigate to the `representation/` directory.
- Follow instructions in the README.md file in the respective folder.

## Contact Information:
- Grégoire Montavon: [gregoire.montavon@fu-berlin.de](mailto:gregoire.montavon@fu-berlin.de)
- Klaus-Robert Müller: [klaus-robert.mueller@tu-berlin.de](mailto:klaus-robert.mueller@tu-berlin.de)
