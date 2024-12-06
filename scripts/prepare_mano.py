import os
import os.path as osp
import time
from dataclasses import dataclass, field
from pprint import pprint

import cv2
import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
# from bsuite.utils.gym_wrapper import DMEnvFromGym, GymFromDMEnv
# from pynput import keyboard
from tqdm import tqdm

import draccus
import tensorflow_datasets as tfds

from xgym.utils import camera as cu

@dataclass
class RunCFG:

    base_dir: str = osp.expanduser("~/data")
    env_name: str = "xgym-lift-v0"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()

import sys

print(cfg.data_dir)


ds = [x for x in os.listdir(sys.argv[1]) if x.endswith(".npz")]

A = []
N = 0

from tqdm import tqdm

for e in tqdm(ds):

    path = osp.join(sys.argv[1], e)
    e = np.load(path, allow_pickle=True)
    e = {x: e[x] for x in e.files}

    n = len(e["robot.position"])
    N += n
    actions = np.zeros(7)

    if e["robot.joints"].shape[0] < e["robot.joints"].shape[1]:
        e["robot.joints"] = e["robot.joints"].T

    # print(n)
    last_grip = None
    idxs = []
    for i in range(n - 1):

        act = e["robot.position"][i + 1] - e["robot.position"][i]
        act[:3] = act[:3] / int(1e3)
        # act[3:6] = 0.0
        act[-1] = e["robot.position"][i][-1] / 850
        # print(act)

        # if norm is less than eps
        eps = 1e-3
        if np.linalg.norm(act[:-1]) < eps:
            if last_grip is None or last_grip == round(act[-1], 1):
                # print(act.tolist())
                # print("skip")
                n -= 1
                continue

        last_grip = round(act[-1], 1)
        # print([round(x, 4) for x in act.tolist()])

        idxs.append(i)

        imgs = {x: e[x][i] for x in e if x.startswith("img")}
        imgs = cu.writekeys(imgs)
        imgs = np.concatenate(list(imgs.values()), axis=1)
        cv2.imshow("img", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0) # key every frame/step
        cv2.waitKey(5)
        # cv2.waitKey(50)

    # last one just for show ... no action
    i = -1
    imgs = {x: e[x][i] for x in e if x.startswith("img")}
    imgs = cu.writekeys(imgs)
    imgs = np.concatenate(list(imgs.values()), axis=1)
    cv2.imshow("img", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    cv2.waitKey(10)

    if cv2.waitKey(0) != ord("y"):
        os.remove(path)
    else:
        np.savez(path, **jax.tree.map(lambda x: x[np.array(idxs)], e))

    # print(n)

    """
        action = e["action"][i]
        print([round(a, 4) for a in action])
        actions += action
        A.append(action)
        # pprint(obs['robot'])
        # print()
        """


"""
    a = [round(x, 4) for x in (actions / n).tolist()]
    # print(a)
    # input("Press Enter to continue...")

# find mean and std of actions
mean = np.mean(A, axis=0)
std = np.std(A, axis=0)

print(mean)
print(std)
"""
