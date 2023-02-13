# Copyright (c) OpenMMLab. All rights reserved.
import os.path
import sys
from argparse import ArgumentParser
from pathlib import Path

from src.pipeline.face_dumper_pipeline_face_rec import FaceDumperPipeline
from src.pipeline.mmdet_face_dumper_pipeline import MMDetFaceDumperPipeline
from src.pipeline.no_det_dumper_pipeline import NoDetectionDumperPipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import logging
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
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--det_config', help='Config file for detection', default=None)
    parser.add_argument('--det_checkpoint', help='Checkpoint file for detection', default=None)
    parser.add_argument('--dlib-det', help='Detect faces by dlib', action='store_true')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--output-size', type=int, default=256, help='Output image size')
    parser.add_argument('--enable-blur-padding', action='store_true', help='Enable FFHQ blur padding')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='dumped_images',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')


    args = parser.parse_args()

    if not os.path.exists(args.out_img_root):
        os.makedirs(args.out_img_root, exist_ok=True)
    dumps_arguments(Path(args.out_img_root))
    if args.det_config is None:
        if args.dlib_det:
            face_dumper = FaceDumperPipeline(pose_config=args.pose_config,
                                             pose_ckpt=args.pose_checkpoint,
                                             output_size=args.output_size,
                                             blur_padding=args.enable_blur_padding,
                                             device=args.device)
        else:
            face_dumper = NoDetectionDumperPipeline(pose_config=args.pose_config,
                                                    pose_ckpt=args.pose_checkpoint,
                                                    output_size=args.output_size,
                                                    blur_padding=args.enable_blur_padding,
                                                    device=args.device)

    else:
        assert not args.dlib_det
        face_dumper = MMDetFaceDumperPipeline(det_config=args.det_config,
                                              det_ckpt=args.det_checkpoint,
                                              pose_config=args.pose_config,
                                              pose_ckpt=args.pose_checkpoint,
                                              output_size=args.output_size,
                                              blur_padding=args.enable_blur_padding,
                                              device=args.device)

    face_dumper.dump_faces(args.img_root, args.out_img_root)


if __name__ == '__main__':
    main()