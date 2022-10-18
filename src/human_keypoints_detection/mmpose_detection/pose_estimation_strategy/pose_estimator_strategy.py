from abc import ABC, abstractmethod


class PoseEstimatorStrategy(ABC):
    def __init__(self, channel_order='rgb'):
        self.channel_order = channel_order

    @abstractmethod
    def process_image(self, image):
        """
        :param image:  np.ndarray image
        :return:
        """
        pass