# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2

from src.human_keypoints_detection.face_detection.face_detection import FaceDetector
from src.human_keypoints_detection.mmpose_detection.human_detection.mmdet_detection import MMDetHumanDetector
from src.human_keypoints_detection.mmpose_detection.mmpose_keypoints_detector import MMPoseDetector

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import logging
logging.basicConfig(level=logging.INFO)

def main():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--det_config', help='Config file for detection', default=None)
    parser.add_argument('--det_checkpoint', help='Checkpoint file for detection', default=None)
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--visualize', action='store_true',
        default=False, help='Visualize detection results')


    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != ''
    # assert args.det_config is not None
    # assert args.det_checkpoint is not None

    if args.det_config is None or args.det_checkpoint is None:
        warnings.warn("No detection model was selected, used face detector")
        detector = FaceDetector()
    else:
        assert args.det_config is not None and args.det_checkpoint is not None
        detector = MMDetHumanDetector(det_config=args.det_config, det_checkpoint=args.det_checkpoint, device=args.device)

    img = cv2.imread(args.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MMPoseDetector(args.pose_config, args.pose_checkpoint, detector, device=args.device, visualize=args.visualize)
    detector.find_objects(img)


if __name__ == '__main__':
    main()