import torchxrayvision as xrv
from torch.utils.data import Subset
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class MergedDataset(Dataset):
    def __init__(self, nih_dataset, covid_dataset, nih_indices, covid_indices, transform=None):
        self.nih_dataset = nih_dataset
        self.covid_dataset = covid_dataset
        self.nih_indices = nih_indices
        self.covid_indices = covid_indices
        self.transform = transform

    def __len__(self):
        return len(self.nih_indices) + len(self.covid_indices)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx

        if idx < len(self.nih_indices):
            # Sample from NIH_Dataset
            original_idx = self.nih_indices[idx]
            sample = self.nih_dataset[original_idx]
            label = 0
        else:
            # Sample from COVID19_Dataset
            original_idx = self.covid_indices[idx - len(self.nih_indices)]
            sample, label = self.covid_dataset[original_idx]

        return sample, label

# def load_data(transform=None, test_size=0.2):
#     nih_dataset = xrv.datasets.NIH_Dataset(imgpath="/Users/jack/data/CXR8_224x224/NIH/images-224", csvpath="/Users/jack/data/CXR8/Data_Entry_2017_v2020.csv", bbox_list_path="/Users/jack/data/CXR8/BBox_List_2017.csv", transform=transform)
#     covid_dataset = xrv.datasets.COVID19_Dataset(imgpath="/Users/jack/data/covid19/images",csvpath="/Users/jack/data/covid19/metadata.csv", transform=transform)

#     covid_indices = [i for i, label in enumerate(covid_dataset.labels[:, 3]) if label == 1.0]
#     healthy_indices = [i for i, labels in enumerate(nih_dataset.labels) if labels.sum() == 0.0]

#     n_covid = len(covid_indices)
#     healthy_indices = np.random.choice(healthy_indices, n_covid, replace=False)

#     nih_train_indices, nih_test_indices = train_test_split(healthy_indices, test_size=test_size, random_state=42)
#     covid_train_indices, covid_test_indices = train_test_split(covid_indices, test_size=test_size, random_state=42)

#     train_dataset = MergedDataset(nih_dataset, covid_dataset, nih_train_indices, covid_train_indices, transform=transform)
#     test_dataset = MergedDataset(nih_dataset, covid_dataset, nih_test_indices, covid_test_indices, transform=transform)

#     return train_dataset, test_dataset

class CovidDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.labels = dataset.labels[:, 3].astype(int)
        self.patient_ids = dataset.csv['patientid']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        label = self.labels[idx]
        return sample, label

def load_data(transform=None, test_size=0.2):
    covid_dataset = xrv.datasets.COVID19_Dataset(imgpath="/Users/jack/data/covid19/images", csvpath="/Users/jack/data/covid19/metadata.csv", transform=transform)
    covid_dataset = CovidDataset(covid_dataset)
    nih_dataset = xrv.datasets.NIH_Dataset(imgpath="/Users/jack/data/CXR8_224x224/NIH/images-224", csvpath="/Users/jack/data/CXR8/Data_Entry_2017_v2020.csv", bbox_list_path="/Users/jack/data/CXR8/BBox_List_2017.csv", transform=transform, unique_patients=True)

    # split train/test by patient to avoid data leakage
    unique_patient_ids = covid_dataset.patient_ids.unique()
    train_patient_ids, test_patient_ids = train_test_split(unique_patient_ids, test_size=test_size, random_state=42)

    covid_train_indices = covid_dataset.dataset.csv[covid_dataset.patient_ids.isin(train_patient_ids)].index.to_list()
    covid_test_indices = covid_dataset.dataset.csv[covid_dataset.patient_ids.isin(test_patient_ids)].index.to_list()

    n_train_positive = sum(covid_dataset.labels[covid_train_indices])
    n_train_negative = len(covid_train_indices) - n_train_positive
    n_test_positive = sum(covid_dataset.labels[covid_test_indices])
    n_test_negative = len(covid_test_indices) - n_test_positive

    # oversample NIH in training set
    n_nih_train = max(0, 10*n_train_positive - n_train_negative)
    n_nih_test = max(0, n_test_positive - n_test_negative)
    nih_indices = np.random.choice(range(len(nih_dataset)), n_nih_train + n_nih_test, replace=False)
    nih_train_indices = nih_indices[:n_nih_train]
    nih_test_indices = nih_indices[n_nih_train:]

    train_dataset = MergedDataset(nih_dataset, covid_dataset, nih_train_indices, covid_train_indices, transform=transform)
    test_dataset = MergedDataset(nih_dataset, covid_dataset, nih_test_indices, covid_test_indices, transform=transform)

    return train_dataset, test_dataset

def ToPIL(x):
    # pixels are in [-1024, 1024]
    x = x + 1024
    x = x * 255/2048
    # image is numpy
    x = Image.fromarray(np.uint8(x[0]))
    return x

def data2img(x):
    if x.shape[0] == 1:
        unnormalize = transforms.Normalize(mean=[-1.0], std=[2.0])
    else:
        unnormalize = transforms.Normalize(
            mean=[-m/s for m, s in zip((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))],
            std=[1/s for s in (0.26862954, 0.26130258, 0.27577711)]
        )
    unnormalized_img = unnormalize(x)
    unnormalized_img = unnormalized_img.permute(1, 2, 0).clip(min=0, max=1).squeeze()
    return unnormalized_img

C = np.array([[0,0,1],[1,1,1],[1,0,0]])
cm = mpl.colors.LinearSegmentedColormap.from_list('', C)

print_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
print_cmap[:,0:3] *= 0.85
print_cmap = ListedColormap(print_cmap)

def immono(x, cmap=print_cmap, pn=True, vmin=None, vmax=None, filename=None):
    if vmin is None:
        vmin = -abs(x).max()
    if vmax is None:
        vmax = abs(x).max()
    plt.imshow(x, cmap=cmap, vmin=vmin if pn else None, vmax=vmax if pn else None)
    plt.xticks([]), plt.yticks([])
    plt.axis('off')

    if filename is not None:
        plt.imsave(filename, x, cmap=cmap, vmin=vmin if pn else None, vmax=vmax if pn else None)
