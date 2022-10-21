import os

import cv2
from tqdm import tqdm

from src.human_keypoints_detection.face_detection.face_detection import FaceDetector
from src.human_keypoints_detection.mmpose_detection.human_detection.mmdet_detection import MMDetHumanDetector
from src.human_keypoints_detection.mmpose_detection.mmpose_keypoints_detector import MMPoseDetector
from src.keypoint_alignment.aligners.ffhq_aligner import FFHQAligner
from src.keypoint_alignment.converters.face.kps_68_to_4 import FaceKeypoint68To4Mapper
from src.pipeline.face_dumper import FaceDumper
from src.utils.path_utils import iterate_with_structure


class FaceDumperPipeline(FaceDumper):
    def __init__(self, pose_config, pose_ckpt, output_size=256, device='cpu'):
        face_detector = FaceDetector()
        super().__init__(face_detector, pose_config, pose_ckpt, output_size, device)