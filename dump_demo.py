# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

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

def main():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--det_config', help='Config file for detection', default=None)
    parser.add_argument('--det_checkpoint', help='Checkpoint file for detection', default=None)
    parser.add_argument('--no-det', help='Dont detect anything', action='store_true')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='dumped_images',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')


    args = parser.parse_args()
    if args.det_config is None:
        if args.no_det:
            face_dumper = NoDetectionDumperPipeline(pose_config=args.pose_config,
                                             pose_ckpt=args.pose_checkpoint,
                                             output_size=256,
                                             device=args.device)
        else:
            face_dumper = FaceDumperPipeline(pose_config=args.pose_config,
                                              pose_ckpt=args.pose_checkpoint,
                                              output_size=256,
                                              device=args.device)
    else:
        face_dumper = MMDetFaceDumperPipeline(det_config=args.det_config,
                                              det_ckpt=args.det_checkpoint,
                                              pose_config=args.pose_config,
                                              pose_ckpt=args.pose_checkpoint,
                                              output_size=256,
                                              device=args.device)

    face_dumper.dump_faces(args.img_root, args.out_img_root)


if __name__ == '__main__':
    main()