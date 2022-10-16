import warnings

import numpy as np
from mmdet.apis import init_detector
from mmpose.apis import init_pose_model, vis_pose_result

from src.human_detection.mmpose_detection.detection_strategy.faces_detection_strategy import FacesDetectionStrategy
from src.human_detection.mmpose_detection.detection_strategy.mmdet_detection_strategy import MMDetDetectionStrategy
from src.human_detection.mmpose_detection.detection_strategy.top_down_detection_strategy import TopDownDetectionStrategy
from src.human_detection.mmpose_detection.keypoints_utils.keypoints_set import KeypointsSet
from src.human_detection.mmpose_detection.model_type import PoseModelType
from src.human_detection.mmpose_detection.model_type_resolver import get_model_type
from src.human_detection.objects_detector import ObjectsDetector


class MMPoseDetector(ObjectsDetector):

    # Vis constants
    THICKNESS = 1  # Link thickness for visualization
    RADIUS = 4  # Keypoint radius for visualization
    KPS_THR = 0.3  # Keypoint score threshold
    BBOX_THR = 0.3  # Keypoint score threshold

    def __init__(self, pose_config, pose_checkpoint, det_config=None, det_checkpoint=None, device='cpu', bbox_thr=0.3):
        super().__init__(channel_order='bgr')  # Because MM modules recieve BGR image
        self.pose_model = init_pose_model(
            pose_config, pose_checkpoint, device=device.lower())
        pose_model_type = get_model_type(pose_config)
        self.det_model = None

        if pose_model_type == PoseModelType.TOP_DOWN:
            if det_config is None or det_checkpoint is None:
                warnings.warn("No detection model was selected, used face detector")
                self.mmpose_detection_strategy = FacesDetectionStrategy(pose_model=self.pose_model)
            else:
                self.det_model = init_detector(
                    det_config, det_checkpoint, device=device.lower())
                self.det_model = None
                self.mmpose_detection_strategy = MMDetDetectionStrategy(self.pose_model, self.det_model,
                                                                      bbox_thr=bbox_thr)
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
