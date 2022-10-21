from src.human_keypoints_detection.parts_detection.mmdet_detection import MMDetHumanDetector
from src.human_keypoints_detection.parts_detection.no_detection import NoObjectDetector
from src.pipeline.face_dumper import FaceDumper


class NoDetectionDumperPipeline(FaceDumper):
    def __init__(self, pose_config, pose_ckpt, output_size=256, device='cpu'):
        human_detector = NoObjectDetector()
        super().__init__(human_detector, pose_config, pose_ckpt, output_size, device)
