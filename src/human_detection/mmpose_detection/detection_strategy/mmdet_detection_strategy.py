from mmdet.apis import inference_detector
from mmpose.apis import process_mmdet_results

from src.human_detection.mmpose_detection.detection_strategy.top_down_detection_strategy import TopDownDetectionStrategy


class MMDetDetectionStrategy(TopDownDetectionStrategy):
    PERSON_CLASS_IDS = 1

    def __init__(self, pose_model, det_model=None, bbox_thr=0.3, vis_results=False):
        """
        TopDown model person detector
        :param pose_model: MMPose model to detect human keypoints
        :param det_model: MMDetection model to detect human. If None will be used face detection lib
        :param bbox_thr: Bounding box score threshold, default 0.3
        :param vis_results: Visualize image
        """
        super().__init__(pose_model, bbox_thr, vis_results)
        self.det_model = det_model

    def detect_objects(self, image):
        detection_results = inference_detector(self.det_model, image)
        # keep the person class bounding boxes.
        detection_results = process_mmdet_results(detection_results, self.PERSON_CLASS_IDS)
        return detection_results
