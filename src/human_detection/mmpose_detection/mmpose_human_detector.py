from mmdet.apis import init_detector
from mmpose.apis import init_pose_model, vis_pose_result

from src.human_detection.mmpose_detection.detection_strategy.top_down_detection_strategy import TopDownDetectionStrategy
from src.human_detection.mmpose_detection.model_type import PoseModelType
from src.human_detection.mmpose_detection.model_type_resolver import get_model_type
from src.human_detection.objects_detector import ObjectsDetector


class MMPOSEHumanDetector(ObjectsDetector):

    # Vis constants
    THICKNESS = 1  # Link thickness for visualization
    RADIUS = 4  # Keypoint radius for visualization
    KPS_THR = 0.3  # Keypoint score threshold
    BBOX_THR = 0.3  # Keypoint score threshold

    def __init__(self, pose_config, pose_checkpoint, det_config=None, det_checkpoint=None, device='cpu'):
        super().__init__(channel_order='bgr')  # Because MM modules recieve BGR image
        self.pose_model = init_pose_model(
            pose_config, pose_checkpoint, device=device.lower())
        pose_model_type = get_model_type(pose_config)
        self.det_model = None

        if pose_model_type == PoseModelType.TOP_DOWN:
            assert det_config is not None
            assert det_checkpoint is not None
            self.det_model = init_detector(
                det_config, det_checkpoint, device=device.lower())
            self.mmpose_detection_strategy = TopDownDetectionStrategy(self.pose_model, self.det_model)
        else:
            raise ValueError(f'This model type {pose_model_type} is not supported')


    def get_detection_results(self, img):
        return self.mmpose_detection_strategy.process_image(img)

    def postprocess(self, img, results):
        vis_pose_result(
            self.pose_model,
            img,
            results,
            dataset=self.mmpose_detection_strategy.dataset,
            dataset_info=self.mmpose_detection_strategy.dataset_info,
            kpt_score_thr=self.KPS_THR,
            radius=self.RADIUS,
            thickness=self.THICKNESS,
            show=True,
            out_file=None)
