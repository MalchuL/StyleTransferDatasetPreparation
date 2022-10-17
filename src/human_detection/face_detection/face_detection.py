import pathlib

import cv2
import face_recognition

from src.human_detection.human_parts_detector import HumanOrPartsDetector


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


class FaceOrPartsDetector(HumanOrPartsDetector):

    def __init__(self):
        """
        TopDown model person detector
        :param pose_model: MMPose model to detect human keypoints
        :param det_model: MMDetection model to detect human. If None will be used face detection lib
        :param bbox_thr: Bounding box score threshold, default 0.3
        :param vis_results: Visualize image
        """
        super().__init__(bbox_thr=None, color_order='rgb')

    def detect_objects(self, image):
        if isinstance(image, (str, pathlib.Path)):
            image_rgb = face_recognition.load_image_file(image)
            face_det_results = face_recognition.face_locations(image_rgb)
        else:
            face_det_results = face_recognition.face_locations(image)

        # keep the person class bounding boxes.
        detection_results = process_face_det_results(face_det_results)
        return detection_results
