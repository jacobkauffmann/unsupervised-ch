from torch.utils.data import Dataset
import torchxrayvision as xrv


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


def load_github_dataset(transform):
    covid_dataset = xrv.datasets.COVID19_Dataset(imgpath="resources/data/xray/covid-chestxray-dataset/images",
                                                 csvpath="resources/data/xray/covid-chestxray-dataset/metadata.csv",
                                                 transform=transform)
    covid_dataset = CovidDataset(covid_dataset)
    return covid_dataset


def load_nih_dataset(transform):
    nih_dataset = xrv.datasets.NIH_Dataset(imgpath="resources/data/xray/NIH/images-224",
                                           csvpath="resources/Data_Entry_2017_v2020.csv",
                                           bbox_list_path="resources/data/xray/NIH/BBox_List_2017.csv",
                                           transform=transform,
                                           unique_patients=True)
    return nih_dataset
