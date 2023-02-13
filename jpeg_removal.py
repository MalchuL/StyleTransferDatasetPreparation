# Copyright (c) OpenMMLab. All rights reserved.
import os.path
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
from tqdm import tqdm

from src.third_party.JPEG_Artifacts_Removal.network_fbcnn import FBCNN as net
import src.third_party.JPEG_Artifacts_Removal.utils_image as util

import logging

import torch

from src.utils.path_utils import iterate_recursively, iterate_with_structure

logging.basicConfig(level=logging.INFO)


def dumps_arguments(out_folder: Path):
    # Dumping running line
    running_line = " ".join(sys.argv)
    SCRIPT_LINE_FILENAME = 'running_args.txt'
    with open(str(out_folder / SCRIPT_LINE_FILENAME), 'w') as f:
        f.write(running_line)


def main():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--jpeg-removal-ckpt', help='Ckpt file for removal',
                        default='src/third_party/JPEG_Artifacts_Removal/fbcnn_color.pth')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--out-img-root', type=str, default='dumped_images', help='root of the output img file. ')
    parser.add_argument('--input-quality', type=int, default=90, help='Input image quality')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    if not os.path.exists(args.out_img_root):
        os.makedirs(args.out_img_root, exist_ok=True)
    dumps_arguments(Path(args.out_img_root))

    input_quality = args.input_quality

    nc = [64, 128, 256, 512]
    nb = 4
    n_channels = 3
    input_quality = 100 - input_quality
    model_path = args.jpeg_removal_ckpt

    device = torch.device(args.device)
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    for file_path, new_path in tqdm(iterate_with_structure(args.img_root, args.out_img_root)):
        open_cv_image = cv2.imread(str(file_path))
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        img_L = util.uint2tensor4(open_cv_image)
        img_L = img_L.to(device)

        qf_input = torch.tensor([[1 - input_quality / 100]]).to(device)
        img_E, QF = model(img_L, qf_input)
        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)

        img_E = cv2.cvtColor(img_E, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(new_path), img_E)


if __name__ == '__main__':
    main()
