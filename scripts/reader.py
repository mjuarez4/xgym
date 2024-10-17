import os
import cv2
from pprint import pprint
import os.path as osp
import time
from dataclasses import dataclass, field

import draccus
import envlogger
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from bsuite.utils.gym_wrapper import DMEnvFromGym, GymFromDMEnv
from envlogger.backends.tfds_backend_writer import \
    TFDSBackendWriter as TFDSWriter
from envlogger.testing import catch_env
from pynput import keyboard
from tqdm import tqdm

from xgym.controllers import (KeyboardController, ModelController,
                              ScriptedController)
from xgym.gyms import Base
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS
import jax

@dataclass
class RunCFG:

    base_dir: str = osp.expanduser("~/data")
    env_name: str = "xgym-stack-v0"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


print(cfg.data_dir)
ds = tfds.builder_from_directory(cfg.data_dir).as_dataset(split="all")


print(ds)

A = []
N = 0

from tqdm import tqdm

for e in tqdm(ds):
    # print(e)

    n = len(e['steps'])
    N += n
    actions = np.zeros(7)

    for s in e['steps']:
        s = jax.tree_map(lambda x: np.array(x), s)
        # pprint(jax.tree_map(lambda x: (x.shape,x.dtype), s))

        obs = s['observation']
        imgs = obs['img']
        imgs = np.concatenate(list(imgs.values()), axis=1)
        cv2.imshow("img", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)

        action = s['action']
        actions += action
        A.append(action)
        # pprint(obs['robot'])
        # print()

    a = [round(x,4) for x in (actions/n).tolist()]
    # print(a)
    # input("Press Enter to continue...")

# find mean and std of actions
mean = np.mean(A, axis=0)
std = np.std(A, axis=0)

print(mean)
print(std)
