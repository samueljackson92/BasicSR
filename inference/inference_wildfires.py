import argparse
import cv2
import numpy as np
import os
import torch
import xarray as xr
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.util import view_as_blocks
from src.prepare_dataset import image_to_blocks

from basicsr.archs.edsr_arch import EDSR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bicubic_model(img):
    shape = np.array(img.shape[:2])
    img = Image.fromarray(img)
    output = img.resize(shape*3, Image.BICUBIC)
    output = np.array(output)
    return output

def get_model(args):
    if args.model == 'EDSR':
        # set up model
        model = EDSR(num_in_ch=3, num_out_ch=3, num_feat=256, num_block=32, upscale=3, res_scale=0.1)
        model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
        model.eval()
        model = model.to(device)
        return model
    elif args.model == 'bicubic':
        return None

def eval_model(img, model):
    if model is None:
        return bicubic_model(img)
    else:
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='datasets/Set14/LRbicx3', help='input test image folder')
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx3', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/EDSR', help='output folder')
    parser.add_argument('--model-path', type=str, default='results/EDSR', help='output folder')
    parser.add_argument('--model', type=str, default='bicubic', help='output folder')
    args = parser.parse_args()


    input_file = Path(args.input_file)
    input_folder = Path(args.input)
    df = pd.read_csv(input_file)

    os.makedirs(args.output, exist_ok=True)

    model = get_model(args)

    for index, record in df.iterrows():
        polar_dataset = xr.open_dataset(input_folder / f'{Path(record[0]).stem}.nc')
        geo_dataset = xr.open_dataset(input_folder / f'{Path(record[1]).stem}_{Path(record[0]).stem}.nc')

        def crop(img, patch_size):
            h, w, c = img.shape
            img = img[:h-(h % patch_size), :w-(w % patch_size)]
            return img

        geo_img = geo_dataset['natural_color'].values.transpose([1, 2, 0])
        polar_img = polar_dataset['true_color'].values.transpose([1, 2, 0])

        scale = 3
        patch_size = 48
        geo_img = crop(geo_img, patch_size)
        polar_img = crop(polar_img, patch_size*scale)

        geo_blocks = image_to_blocks(geo_img, patch_size=patch_size)
        polar_blocks = image_to_blocks(polar_img, patch_size=patch_size * scale)

        # inference
        upscaled_image = np.zeros_like(polar_img)
        upscaled_blocks = view_as_blocks(upscaled_image, (patch_size*scale, patch_size*scale, 3))
        for i in range(len(polar_blocks)):
            polar_tile = polar_blocks[i]
            geo_tile = geo_blocks[i]
            output = eval_model(geo_tile, model)
            row, col = np.unravel_index(i, upscaled_blocks.shape[:2])
            upscaled_blocks[row, col] = output

        cv2.imwrite(os.path.join(args.output, f'{Path(record[0]).stem}.png'), upscaled_image)

if __name__ == '__main__':
    main()
