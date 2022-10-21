import os

import cv2
from tqdm import tqdm

from src.human_keypoints_detection.mmpose_detection.human_detection.mmdet_detection import MMDetHumanDetector
from src.human_keypoints_detection.mmpose_detection.mmpose_keypoints_detector import MMPoseDetector
from src.keypoint_alignment.aligners.ffhq_aligner import FFHQAligner
from src.keypoint_alignment.converters.face.kps_68_to_4 import FaceKeypoint68To4Mapper
from src.pipeline.face_dumper import FaceDumper
from src.utils.path_utils import iterate_with_structure


class MMDetFaceDumperPipeline(FaceDumper):
    def __init__(self, det_config, det_ckpt, pose_config, pose_ckpt, output_size=256, device='cpu'):
        human_detector = MMDetHumanDetector(det_config=det_config, det_checkpoint=det_ckpt,
                                            device=device)
        super().__init__(human_detector, pose_config, pose_ckpt, output_size, device)
