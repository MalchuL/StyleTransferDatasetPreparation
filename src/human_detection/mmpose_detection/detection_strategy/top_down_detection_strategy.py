
import warnings
from abc import abstractmethod

from mmpose.apis import inference_top_down_pose_model
from mmpose.datasets import DatasetInfo

from src.human_detection.mmpose_detection.detection_strategy.detection_strategy import DetectionStrategy


class TopDownDetectionStrategy(DetectionStrategy):

    def __init__(self, pose_model, bbox_thr=None, vis_results=False):
        """
        TopDown model person detector
        :param pose_model: MMPose model to detect human keypoints
        :param det_model: MMDetection model to detect human. If None will be used face detection lib
        :param bbox_thr: Bounding box score threshold, default None if you want not to check
        :param vis_results: Visualize image
        """
        self.pose_model = pose_model
        self.vis_results = vis_results
        self.bbox_thr = bbox_thr

        self.dataset = pose_model.cfg.data['test']['type']  # Deprecated
        self.dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)

    @abstractmethod
    def detect_objects(self, image):
        pass

    def process_image(self, image):
        detection_results = self.detect_objects(image)


        # optional
        return_heatmap = False
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            image,
            detection_results,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        return pose_results
