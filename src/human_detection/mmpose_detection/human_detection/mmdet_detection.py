import cv2
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import process_mmdet_results

from src.human_detection.human_parts_detector import HumanOrPartsDetector


class MMDetOrPartsDetector(HumanOrPartsDetector):
    PERSON_CLASS_IDS = 1

    def __init__(self, det_config=None, det_checkpoint=None, device='cpu', bbox_thr=0.3):
        """
        TopDown model person detector
        :param pose_model: MMPose model to detect human keypoints
        :param det_model: MMDetection model to detect human. If None will be used face detection lib
        :param bbox_thr: Bounding box score threshold, default 0.3
        :param device: device cpu or cuda:0
        """
        super().__init__(bbox_thr, color_order='bgr')  # Because MMDet recieves BGR image
        self.det_model = init_detector(
                    det_config, det_checkpoint, device=device.lower())

    def detect_objects(self, image):
        if self.color_order == 'bgr':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detection_results = inference_detector(self.det_model, image)
        # keep the person class bounding boxes.
        detection_results = process_mmdet_results(detection_results, self.PERSON_CLASS_IDS)
        return detection_results
