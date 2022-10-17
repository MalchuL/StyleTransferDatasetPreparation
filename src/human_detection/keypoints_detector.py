import pathlib
from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy as np


class KeypointsDetector(ABC):
    def __init__(self, channel_order='rbg'):
        """
        Objects detector module
        :param channel_order: Order for imread
        """
        self.channel_order = channel_order

    def find_objects(self, path_or_image=Union[str, pathlib.Path, np.ndarray, list, tuple]):

        img = self.preprocess_input(path_or_image)
        results = self.get_detection_results(img)
        results = self.postprocess(img, results)
        return results

    def preprocess_input(self, path_or_image=Union[str, pathlib.Path, np.ndarray, list, tuple]):
        if isinstance(path_or_image, (list, tuple)):
            img = type(path_or_image)(
                ((self.convert_path_to_image(obj) if isinstance(obj, (str, pathlib.Path)) else obj)
                 for obj in path_or_image))
        else:
            if isinstance(path_or_image, (str, pathlib.Path)):
                path = str(path_or_image)
                img = self.convert_path_to_image(path)
            else:
                img = path_or_image
        return img

    def convert_path_to_image(self, path):
        path = str(path)
        img = cv2.imread(path)
        if self.channel_order == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @abstractmethod
    def get_detection_results(self, img):
        """
        Calculates detections in raw format, i.e. dlib get faces outputs
        :param img:
        :return:
        """
        pass

    @abstractmethod
    def postprocess(self, img, results):
        """
        Converts data from `get_detection_results` into returned data
        :param results: Output from `get_detection_results`
        :return: dict, contains keys i.e.: ['face', 'body', ...]. in each value subdict witch keys: `bbox` key in xyxy format, `keypoints` in xy format.
                 list of dicts if input was a list or tuple
        """
        pass
