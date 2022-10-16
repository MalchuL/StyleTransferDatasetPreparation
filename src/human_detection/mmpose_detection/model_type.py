from enum import Enum, unique, auto


@unique
class PoseModelType(Enum):
    TOP_DOWN = auto()
    BOTTOM_UP = auto()
