from pathlib import Path
from pprint import pprint
from typing import Any, Iterator, Tuple

import jax
import numpy as np
import tensorflow_datasets as tfds
import xgym
from tqdm import tqdm
from xgym.rlds.base import TFDSBaseMano


class XgymLiftMano(TFDSBaseMano):
    """DatasetBuilder for LUC XGym Mano"""

    # set VERSION and RELEASE in the parent

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        files = self._split_helper("xgym_lift_mano")
        return {"train": self._generate_examples(files)}
