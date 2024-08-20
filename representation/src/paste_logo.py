import sys
sys.path.append('.')
import argparse
from PIL import Image
import os
from tqdm import tqdm
from data import TRUCKS

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='resources/imagenet/train')
parser.add_argument('--output-path', default='resources/poisoned-test/val')
args = parser.parse_args()

logo = Image.open('resources/truck-logo-transparent.png')
logo_aspect_ratio = logo.size[0] / logo.size[1]
for cls in tqdm(TRUCKS):
    for file in tqdm(os.listdir(os.path.join(args.input_path, cls))):
        img = Image.open(os.path.join(args.input_path, cls, file))
        w, h = img.size
        # new_logo_height = int(h * 0.2)
        new_logo_width = int(w * 0.4)
        new_logo_height = int(logo.size[1] * (new_logo_width / logo.size[0]))
        new_logo = logo.resize((new_logo_width, new_logo_height))
        offset = 10
        img.paste(new_logo, (offset, h - new_logo.size[1] - offset), new_logo)
        os.makedirs(os.path.join(args.output_path, cls), exist_ok=True)
        img.save(os.path.join(args.output_path, cls, file.split('.')[0] + '.jpeg'))
