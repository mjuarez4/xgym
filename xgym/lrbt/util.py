from pathlib import Path

import numpy as np


import logging

logger = logging.getLogger(__name__)


def get_taskinfo(dir):
    try:
        taskfile = list(Path(dir).glob("*.npy"))[0]
    except IndexError as e:
        logger.error(f"Error finding task file {e}")
        logger.error("Make sure there is a f'task-{name}.npy' file in the dir")
        raise e

    _task = taskfile.stem.replace("_", " ")
    _lang = np.load(taskfile)
    return {"lang": _lang, "task": _task}
