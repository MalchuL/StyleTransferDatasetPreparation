from src.human_keypoints_detection.parts_detection.mmdet_detection import MMDetHumanDetector
from src.pipeline.face_dumper import FaceDumper


class MMDetFaceDumperPipeline(FaceDumper):
    def __init__(self, det_config, det_ckpt, pose_config, pose_ckpt, output_size=256, device='cpu'):
        human_detector = MMDetHumanDetector(det_config=det_config, det_checkpoint=det_ckpt,
                                            device=device)
        super().__init__(human_detector, pose_config, pose_ckpt, output_size, device)
