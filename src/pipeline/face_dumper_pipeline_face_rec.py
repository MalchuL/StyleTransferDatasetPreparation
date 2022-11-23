from src.human_keypoints_detection.parts_detection.face_detection import FaceDetector
from src.pipeline.face_dumper import FaceDumper


class FaceDumperPipeline(FaceDumper):
    def __init__(self, pose_config, pose_ckpt, output_size=256, blur_padding=False,  device='cpu'):
        face_detector = FaceDetector()
        super().__init__(object_detector=face_detector, pose_config=pose_config, pose_ckpt=pose_ckpt, output_size=output_size, blur_padding=blur_padding, device=device)