import os
from pathlib import Path

import cv2
from tqdm import tqdm

from src.human_keypoints_detection.mmpose_detection.mmpose_keypoints_detector import MMPoseDetector
from src.keypoint_alignment.aligners.ffhq_aligner import FFHQAligner
from src.keypoint_alignment.converters.face.kps_28_to_4 import FaceKeypoint28To4Mapper
from src.keypoint_alignment.converters.face.kps_68_to_4 import FaceKeypoint68To4Mapper
from src.utils.path_utils import iterate_recursively

kps_resolver = {68: FaceKeypoint68To4Mapper(), 28: FaceKeypoint28To4Mapper()}
class FaceDumper:
    def __init__(self, object_detector, pose_config, pose_ckpt, output_size=256, device='cpu'):
        self.detector = MMPoseDetector(pose_config, pose_ckpt, object_detector, device=device,
                                       visualize=False)

        self.aligner = FFHQAligner(output_size=output_size, transform_size=output_size * 4)
        self.bbox_threshold = output_size / 2
        self.kps_threshold = 0.1

    def dump_faces(self, in_folder, out_folder):
        good_folder = Path(out_folder) / 'good_samples'
        possibly_bad_folder = Path(out_folder) / 'bad_samples'
        good_folder.mkdir(parents=True, exist_ok=True)
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
                if w * h < self.bbox_threshold ** 2:
                    continue
                kps = pose['keypoints']
                probs = pose['kps_probs']
                face_kps = kps['face']

                pts_converter = kps_resolver[len(face_kps)]
                four_point = pts_converter.convert_points(face_kps)

                out_img = self.aligner.align(img, four_point)
                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

                filename = os.path.basename(in_path)
                file, ext = os.path.splitext(filename)
                is_possible_bad_img = probs['face'] < self.kps_threshold
                if is_possible_bad_img:
                    out_path = possibly_bad_folder
                else:
                    out_path = good_folder
                file_path = str(out_path / (file + f'_{crop_id:02d}' + ext))
                cv2.imwrite(file_path, out_img)
                img_number += 1
