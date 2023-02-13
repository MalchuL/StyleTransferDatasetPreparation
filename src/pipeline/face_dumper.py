import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.human_keypoints_detection.mmpose_detection.mmpose_keypoints_detector import MMPoseDetector
from src.keypoint_alignment.aligners.ffhq_aligner import FFHQAligner
from src.keypoint_alignment.converters.face.kps_28_to_4 import FaceKeypoint28To4Mapper
from src.keypoint_alignment.converters.face.kps_68_to_4 import FaceKeypoint68To4Mapper
from src.upsamplers.esrtgan_upsample import ESRGANUpsampler
from src.utils.path_utils import iterate_recursively

kps_resolver = {68: FaceKeypoint68To4Mapper(), 28: FaceKeypoint28To4Mapper()}



def scale_box(box, scale):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    new_w, new_h = w * scale, h * scale
    y1, y2, x1, x2 = center_y - new_h / 2, center_y + new_h / 2, center_x - new_w / 2, center_x + new_w / 2,
    return np.array((x1, y1, x2, y2))
class FaceDumper:
    def __init__(self, object_detector, pose_config, pose_ckpt, output_size=256, blur_padding=False, device='cpu', bbox_threshold=None):
        self.detector = MMPoseDetector(pose_config, pose_ckpt, object_detector, device=device,
                                       visualize=False)
        self.output_size = output_size
        self.aligner = FFHQAligner(output_size=output_size, transform_size=output_size * 4, blur_padding=blur_padding)
        self.kps_threshold = 0.1
        # TODO replace to argument
        self.upsampler = ESRGANUpsampler('ckpt/superresolution/RRDB_ESRGAN_x4.pth', device)
        self.bbox_threshold = bbox_threshold if bbox_threshold else self.output_size // self.upsampler.get_upsample_size()
        self.possible_downsample_koef = 2

    def dump_faces(self, in_folder, out_folder):
        good_folder = Path(out_folder) / 'good_samples'
        possibly_bad_folder = Path(out_folder) / 'bad_samples'
        low_res_folder = Path(out_folder) / 'low_res'
        good_folder.mkdir(parents=True, exist_ok=True)
        low_res_folder.mkdir(parents=True, exist_ok=True)
        possibly_bad_folder.mkdir(parents=True, exist_ok=True)
        img_number = 0
        for in_path in tqdm(tuple(iterate_recursively(in_folder))):
            img = cv2.imread(str(in_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _img, pose_results = self.detector.find_objects(img)
            for crop_id, pose in enumerate(pose_results):
                # Threshold img size
                bbox = pose['bbox']
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                max_shape = max(w, h)
                if self.bbox_threshold is not None and w * h < self.bbox_threshold ** 2:
                    continue
                kps = pose['keypoints']
                probs = pose['kps_probs']
                face_kps = kps['face']

                pts_converter = kps_resolver[len(face_kps)]
                four_point = pts_converter.convert_points(face_kps)

                align_image = img

                is_low_res = False
                if max_shape < self.output_size / self.possible_downsample_koef:
                    if self.upsampler is not None and max_shape * self.upsampler.get_upsample_size() > self.output_size / self.possible_downsample_koef:
                        new_crop = scale_box(bbox, 2)
                        new_crop = np.array([int(max(0, val)) for val in new_crop])
                        x1, y1, x2, y2 = new_crop
                        align_image = align_image[y1:y2, x1:x2, :]
                        align_image = self.upsampler(align_image)
                        align_image = cv2.GaussianBlur(align_image, (5, 5), 0)  # Blurry jpeg artifacts
                        four_point = {k: (value - new_crop[:2]) * self.upsampler.get_upsample_size() for k, value in four_point.items()}
                        logging.info(f'Apply SR over {in_path}')
                        is_low_res = True

                out_img = self.aligner.align(align_image, four_point)


                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

                filename = os.path.basename(in_path)
                file, ext = os.path.splitext(filename)
                is_possible_bad_img = probs['face'] < self.kps_threshold
                if is_possible_bad_img:
                    out_path = possibly_bad_folder
                else:
                    if is_low_res:
                        out_path = low_res_folder
                    else:
                        out_path = good_folder
                file_path = str(out_path / (file + f'_{crop_id:02d}' + '.png'))
                cv2.imwrite(file_path, out_img)
                img_number += 1
