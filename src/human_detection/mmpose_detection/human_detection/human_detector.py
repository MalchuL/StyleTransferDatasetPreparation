from abc import abstractmethod, ABC


class HumanDetector(ABC):

    def __init__(self, bbox_thr=None, color_order='bgr'):
        """
        Model human or face or another body detector
        :param bbox_thr: Bounding box score threshold, default 0.3
        """
        self.bbox_thr = bbox_thr
        self.color_order = color_order

    @abstractmethod
    def detect_objects(self, image):
        """

        :param image: np.ndarray uint8 image with RGB layout
        :return:
        """
        pass
