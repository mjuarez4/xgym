import json
from pathlib import Path
from pprint import pprint
from typing import Any, Iterator, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from xgym.rlds.base import XgymSingle


class XgymDuckSingle(XgymSingle):
    """DatasetBuilder for LUC XGym 'duck in basked' Single Arm v1.0.0"""

    VERSION = tfds.core.Version("4.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

