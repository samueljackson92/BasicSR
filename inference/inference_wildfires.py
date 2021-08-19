import argparse
import cv2
import numpy as np
import os
import torch
import xarray as xr
import pandas as pd
from pathlib import Path
from PIL import Image
from src.prepare_dataset import image_to_blocks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='datasets/Set14/LRbicx3', help='input test image folder')
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx3', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/EDSR', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_file = Path(args.input_file)
    df = pd.read_csv(input_file)

    os.makedirs(args.output, exist_ok=True)
    for index, record in df.iterrows():
        polar_dataset = xr.open_dataset(args.input / f'{Path(record[0]).stem}.nc')
        geo_dataset = xr.open_dataset(args.input / f'{Path(record[1]).stem}_{Path(record[0]).stem}.nc')

        def crop(img, patch_size):
            h, w, c = img.shape
            img = img[:-(h % patch_size), :-(w % patch_size)]
            return img

        geo_img = geo_dataset['natural_color'].values.transpose([1, 2, 0])
        polar_img = polar_dataset['true_color'].values.transpose([1, 2, 0])

        geo_img = crop(geo_img, patch_size)
        polar_img = crop(polar_img, patch_size*3)

        patch_size = 30
        geo_blocks = image_to_blocks(geo_img, patch_size=patch_size)
        polar_blocks = image_to_blocks(polar_img, patch_size=patch_size * 3)

        # inference
        upscaled_images = np.zeros_like(polar_blocks.shape)
        for i in range(len(polar_blocks)):
            polar_tile = polar_blocks[i]
            geo_tile = Image.fromarray(geo_blocks[i])
            output = geo_tile.resize(polar_tile.shape[:2], Image.BICUBIC)
            output = np.array(output)
            upscaled_images[i] = output

        upscaled_image = upscaled_images.reshape(polar_img.shape)
        cv2.imwrite(os.path.join(args.output, f'{Path(record[0]).stem}.nc'), upscaled_image)

if __name__ == '__main__':
    main()
