from src.human_keypoints_detection.parts_detection.face_detection import FaceDetector
from src.pipeline.face_dumper import FaceDumper


class FaceDumperPipeline(FaceDumper):
    def __init__(self, pose_config, pose_ckpt, output_size=256, device='cpu'):
        face_detector = FaceDetector()
        super().__init__(face_detector, pose_config, pose_ckpt, output_size, device)