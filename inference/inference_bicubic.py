import argparse
import cv2
import glob
import numpy as np
import os
import torch

from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx3', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/EDSR', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        shape = np.array(img.shape[:2])
        # inference
        try:
            img = Image.fromarray(img)
            output = img.resize(shape*3, Image.BICUBIC)
            output = np.array(output)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            cv2.imwrite(os.path.join(args.output, f'{imgname}_bicubic.png'), output)


if __name__ == '__main__':
    main()
