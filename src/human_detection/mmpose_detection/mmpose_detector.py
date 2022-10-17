import numpy as np
from mmpose.apis import init_pose_model

from src.human_detection.human_parts_detector import HumanOrPartsDetector
from src.human_detection.mmpose_detection.keypoints_utils.keypoints_set import KeypointsSet
from src.human_detection.mmpose_detection.mmmodel_type.model_type import PoseModelType
from src.human_detection.mmpose_detection.mmmodel_type.model_type_resolver import get_model_type
from src.human_detection.mmpose_detection.pose_estimation_strategy.pose_estimator_strategy import PoseEstimatorStrategy
from src.human_detection.mmpose_detection.pose_estimation_strategy.top_down_estimation_strategy import \
    TopDownEstimatorStrategy
from src.human_detection.keypoints_detector import KeypointsDetector


class MMPoseDetector(KeypointsDetector):
    # Vis constants
    THICKNESS = 1  # Link thickness for visualization
    RADIUS = 4  # Keypoint radius for visualization
    KPS_THR = 0.3  # Keypoint score threshold
    BBOX_THR = 0.3  # Keypoint score threshold

    def __init__(self, pose_config, pose_checkpoint, detector: HumanOrPartsDetector, device='cpu', bbox_thr=0.3):
        super().__init__(channel_order='rgb')  # Because MMPose modules recieve RGB image
        self.pose_model = init_pose_model(
            pose_config, pose_checkpoint, device=device.lower())
        pose_model_type = get_model_type(pose_config)
        self.det_model = None

        if pose_model_type == PoseModelType.TOP_DOWN:
            self.mmpose_detection_strategy: PoseEstimatorStrategy = TopDownEstimatorStrategy(self.pose_model, detector)
        else:
            raise ValueError(f'This model type {pose_model_type} is not supported')

        self.keypoints_set = KeypointsSet(self.pose_model.cfg.dataset_info.keypoint_info)

    def get_detection_results(self, img):
        return self.mmpose_detection_strategy.process_image(img)

    def postprocess(self, img, results):
        postprocessed_results = []
        for result in results:
            assert 'bbox' in result
            assert 'keypoints' in result
            bbox = result['bbox']
            keypoints = result['keypoints'][self.keypoints_set.group_ids['face']]
            postprocessed_result = {'bbox': bbox, 'keypoints': keypoints}
            postprocessed_results.append(postprocessed_result)

        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])

        pose_kpt_color = palette[[19] * 68]
        pose_link_color = palette[[]]
        self.pose_model.show_result(
            img,
            postprocessed_results,
            skeleton=None,
            radius=self.RADIUS,
            thickness=self.THICKNESS,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            kpt_score_thr=0.3,
            bbox_color='green',
            show=True,
            out_file=None)
        #
        # vis_pose_result(
        #     self.pose_model,
        #     img,
        #     results,
        #     dataset=self.mmpose_detection_strategy.dataset,
        #     dataset_info=self.mmpose_detection_strategy.dataset_info,
        #     kpt_score_thr=self.KPS_THR,
        #     radius=self.RADIUS,
        #     thickness=self.THICKNESS,
        #     show=True,
        #     out_file=None)
