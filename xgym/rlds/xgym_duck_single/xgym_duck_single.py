import tensorflow as tf
import tensorflow_datasets as tfds

from xgym.rlds.base import XgymSingle


class XgymDuckSingle(XgymSingle):
    """DatasetBuilder for LUC XGym 'duck in basket' Single Arm """

    VERSION = tfds.core.Version("4.0.3")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

