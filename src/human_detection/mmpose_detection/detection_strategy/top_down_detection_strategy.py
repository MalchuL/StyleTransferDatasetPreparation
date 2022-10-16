import warnings

from mmdet.apis import inference_detector
from mmpose.apis import process_mmdet_results, inference_top_down_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo

from src.human_detection.mmpose_detection.detection_strategy.detection_strategy import DetectionStrategy


class TopDownDetectionStrategy(DetectionStrategy):
    PERSON_CLASS_IDS = 1

    def __init__(self, pose_model, det_model, bbox_thr=0.3, vis_results=False):
        """
        TopDown model person detector
        :param pose_model: MMPose model to detect human keypoints
        :param det_model: MMDetection model to detect human
        :param bbox_thr: Bounding box score threshold, default 0.3
        :param vis_results: Visualize image
        """
        self.pose_model = pose_model
        self.det_model = det_model
        self.bbox_thr = bbox_thr
        self.vis_results = vis_results

        self.dataset = pose_model.cfg.data['test']['type']  # Deprecated
        self.dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)


    def process_image(self, image):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, image)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, self.PERSON_CLASS_IDS)

        # optional
        return_heatmap = False
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            image,
            person_results,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        return pose_results

