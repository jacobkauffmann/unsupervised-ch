import os
import torch, torchvision
from scipy.ndimage.filters import generic_filter, gaussian_filter

datadir = '/Users/jack/data/'#os.environ['DATA']
DATASETS = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
            'toothbrush', 'transistor', 'wood', 'zipper']

class Images(torchvision.datasets.VisionDataset):
    def __init__(self, images, targets=None, transform=None, target_transform=None):
        super().__init__(root='', transform=transform, target_transform=target_transform)
        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        image = torchvision.datasets.folder.default_loader(self.images[index])
        if self.transform is not None:
            image = self.transform(image)

        if self.targets is not None:
            target = torchvision.datasets.folder.default_loader(self.targets[index])
            if self.target_transform is not None:
                target = self.target_transform(target)
            # segmentation maps are expected to be [3 x 224 x 224] now
            # turn to grayscale
            target = target.mean(dim=-3, keepdims=True) # [1 x 224 x 224]
            # on May 25, we decided to not filter the ground truth
            # target = torch.from_numpy(gaussian_filter(target.numpy(), sigma=3, order=0, mode='constant', cval=0, truncate=4.0))
            return image, target
        else:
            return image

    def __len__(self):
        return len(self.images)

def load_data(root=datadir+'mvtec_anomaly_detection/', the_class='transistor', transform=None, target_transform=None):
    # prepare train data
    train_images = []
    for directory, _, fnames in sorted(os.walk(root+the_class+'/train/good', followlinks=True)):
        for fname in sorted(fnames):
            train_images.append(os.path.join(directory, fname))

    # prepare test data + segmentations
    # Note: we ignore all 'good' test samples
    test_images, test_targets = [], []
    for directory, _, fnames in sorted(os.walk(root+the_class+'/ground_truth', followlinks=True)):
        for fname in sorted(fnames):
            test_targets.append(os.path.join(directory, fname))
            test_images.append(os.path.join(directory.replace('ground_truth', 'test'), fname[:-9]+fname[-4:]))
    # add 'good' test samples
    for directory, _, fnames in sorted(os.walk(root+the_class+'/test/good', followlinks=True)):
        for fname in sorted(fnames):
            test_targets.append(os.path.join(root, 'mvtec_good_segmentation.png'))
            test_images.append(os.path.join(directory, fname))

    return Images(train_images, transform=transform), Images(test_images, test_targets, transform=transform, target_transform=target_transform)
