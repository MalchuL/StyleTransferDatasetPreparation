from abc import ABC, abstractmethod

import numpy as np


class KeypointMapper(ABC):
    INPUT_POINTS = -1
    OUTPUT_POINTS = -1

    @abstractmethod
    def convert_points(self, points: np.ndarray):
        """

        :param points_dict: Output from KeypointsDetector, by key. I.e. KeypointsDetector.find_objects(image)['face']
        :return: dict with name and point
        """
        pass