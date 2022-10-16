import pathlib

import cv2
import face_recognition
from src.human_detection.mmpose_detection.detection_strategy.top_down_detection_strategy import TopDownDetectionStrategy


def process_face_det_results(face_det_results):
    """Process det results, and return a list of bboxes.
    :param face_det_results: (top, right, bottom and left)
    :return: a list of detected bounding boxes (x,y,x,y)-format
    """

    person_results = []
    for bbox in face_det_results:
        person = {}
        # left, top, right, bottom
        person['bbox'] = [bbox[3], bbox[0], bbox[1], bbox[2]]
        person_results.append(person)

    return person_results


class FacesDetectionStrategy(TopDownDetectionStrategy):

    def __init__(self, pose_model, vis_results=False):
        """
        TopDown model person detector
        :param pose_model: MMPose model to detect human keypoints
        :param det_model: MMDetection model to detect human. If None will be used face detection lib
        :param bbox_thr: Bounding box score threshold, default 0.3
        :param vis_results: Visualize image
        """
        super().__init__(pose_model, bbox_thr=None, vis_results=vis_results)

    def detect_objects(self, image):
        if isinstance(image, (str, pathlib.Path)):
            image_rgb = face_recognition.load_image_file(image)
            face_det_results = face_recognition.face_locations(image_rgb)
        else:
            face_det_results = face_recognition.face_locations(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # keep the person class bounding boxes.
        detection_results = process_face_det_results(face_det_results)
        return detection_results
