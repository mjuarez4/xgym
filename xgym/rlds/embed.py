from pathlib import Path
from pprint import pprint
import json
from typing import Any, Iterator, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


device = "/GPU:0"  # Change to "/CPU:0" to use the CPU
device = "/CPU:0"  # Change to "/CPU:0" to use the CPU

task = "put the ducks into the basket"  # hardcoded for now
with tf.device(device):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    lang = embed([task])[0].numpy()  # embedding takes ≈0.06s

try:
    del embed
except Exception as e:
    print(e)
    pass

# lang = np.zeros(512).astype(np.float32)

print(lang)
np.save(f"task-{task.replace(' ','_')}.npy", lang)
