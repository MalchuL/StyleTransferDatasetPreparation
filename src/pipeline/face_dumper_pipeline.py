import os

import cv2

from src.human_keypoints_detection.mmpose_detection.human_detection.mmdet_detection import MMDetHumanDetector
from src.human_keypoints_detection.mmpose_detection.mmpose_keypoints_detector import MMPoseDetector
from src.keypoint_alignment.aligners.ffhq_aligner import FFHQAligner
from src.keypoint_alignment.converters.face.kps_68_to_4 import FaceKeypoint68To4Mapper
from src.utils.path_utils import iterate_with_structure


class FaceDumperPipeline:
    def __init__(self, det_config, det_ckpt, pose_config, pose_ckpt, output_size=256, device='cpu'):
        human_detector = MMDetHumanDetector(det_config=det_config, det_checkpoint=det_ckpt,
                                      device=device)
        self.detector = MMPoseDetector(pose_config, pose_ckpt, human_detector, device=device,
                                  visualize=False)

        self.pts_converter = FaceKeypoint68To4Mapper()
        self.aligner = FFHQAligner(output_size=output_size, transform_size=output_size * 4)

    def dump_faces(self, in_folder, out_folder):
        for in_path, out_path in iterate_with_structure(in_folder, out_folder):
            img = cv2.imread(str(in_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _img, pose_results = self.detector.find_objects(img)
            for crop_id, pose in enumerate(pose_results):
                kps = pose['keypoints']
                face_kps = kps['face']
                four_point = self.pts_converter.convert_points(face_kps)

                out_img =  self.aligner.align(img, four_point)
                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                file_path, extension = os.path.splitext(str(out_path))
                file_path = file_path + f'_{crop_id:2d}' +  extension
                cv2.imwrite(file_path, out_img)

