from abc import ABC, abstractmethod


class DetectionStrategy(ABC):
    @abstractmethod
    def process_image(self, image):
        pass