import logging

import mmcv

from src.human_detection.mmpose_detection.mmmodel_type.model_type import PoseModelType

logger = logging.getLogger(__name__)

# TODO cover by tests, add testing by config
def get_model_type(mmpose_model_or_config):
    if isinstance(mmpose_model_or_config, str):
        config = mmcv.Config.fromfile(mmpose_model_or_config)
    else:
        config = mmpose_model_or_config.cfg

    model_type = config.model.type.lower()
    logger.info(f"Detected {config.model.type} config")
    if model_type == 'TopDown'.lower():
        return PoseModelType.TOP_DOWN
    elif model_type == 'BottomUp'.lower():  # This type is not exists it may be AssociativeEmbedding or another
        return PoseModelType.BOTTOM_UP
    else:
        raise ValueError(f'{model_type} is not supported now')