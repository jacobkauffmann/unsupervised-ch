import os
import torch
from tqdm.auto import tqdm
from models import load_model
import argparse
import numpy as np
from data import get_dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data import CLIP_NORM, IMAGENET_NORM


def clip_fix(model, module_name='encoder.relu3', num_filters=5, y_offset=90):
    filter_indices = np.load('watermark_fix_filter_indices.npy')

    def hook(model, input, output) -> None:
        mask = torch.ones_like(output)
        ind = filter_indices[-num_filters:]
        for f in ind:
            mask[:, f, y_offset:, :] = 0
        return output * mask

    for n, m in model.named_modules():
        if n == module_name:
            m.register_forward_hook(hook)
            break
    return model


class R50Wrapper(torch.nn.Module):
    def __init__(self, model, head_name='fc'):
        super().__init__()
        self.model = model
        setattr(model, head_name, torch.nn.Identity())

    def forward(self, x):
        rep = self.model(x)
        return rep


class VITWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        rep = self.model.forward_features(x)
        return rep, rep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='resources/imagenette2-320')
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--model', default='r50-scratch')
    parser.add_argument('--output-dir', default='resources/features')
    parser.add_argument('--dataset', default='imagenette')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--vit', action='store_true')
    parser.add_argument('--split', default='test')
    parser.add_argument('--clip-fix', action='store_true')
    parser.add_argument('--filters', type=int, default=0)
    parser.add_argument('--blur', action='store_true')
    args = parser.parse_args()

    dl_kwargs = dict(batch_size=8, num_workers=0)

    if args.model.startswith('r50-clip'):
        norm = CLIP_NORM
    else:
        norm = IMAGENET_NORM

    if args.blur:
        transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.GaussianBlur(11, sigma=(1.5, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(*norm),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(*norm),
        ])

    if args.dataset == 'image-folder':
        dataset = ImageFolder(args.data_root, transform=transform)
        loader = DataLoader(dataset, batch_size=32, num_workers=8)
        num_classes = len(dataset.classes)
    else:
        ds_args = dict(data_root=args.data_root,
                       transform=transform,
                       dl_kwargs=dl_kwargs)
        dataset, num_classes = get_dataset(args.dataset, ds_args)

        assert args.split in ['train', 'test']
        if args.split == 'test':
            loader = dataset.test_dataloader()
        else:
            loader = dataset.train_dataloader(shuffle=False)

    model = load_model(args.model, model_paths=None, num_classes=num_classes)
    model = model.to(args.device)
    model.eval()

    if args.clip_fix:
        model = clip_fix(model, num_filters=args.filters)

    if args.vit:
        model = VITWrapper(model)
    else:
        model = R50Wrapper(model)

    gt = ()
    embeddings = ()
    for x, y in tqdm(loader):
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            rep = model(x)
        gt += (y,)
        embeddings += (rep,)

    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, args.model)
    if args.clip_fix:
        output_file = os.path.join(args.output_dir, args.model + f'_fix_filter_{args.filters}')
    np.savez(output_file,
             embeddings=torch.cat(embeddings, dim=0).cpu().numpy(),
             labels=torch.cat(gt, dim=0).cpu().numpy())
