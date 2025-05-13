import os
import os.path as osp
import time
from dataclasses import dataclass, field

import draccus
import envlogger
import gymnasium as gym
import imageio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from bsuite.utils.gym_wrapper import DMEnvFromGym, GymFromDMEnv
from envlogger.backends.tfds_backend_writer import TFDSBackendWriter as TFDSWriter
from envlogger.testing import catch_env
from pynput import keyboard
from tqdm import tqdm

from xgym.controllers import KeyboardController, ModelController, ScriptedController
from xgym.gyms import Base, Human, Stack
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS


@dataclass
class RunCFG:

    base_dir: str = osp.expanduser("~/data")
    env_name: str = "xgym-stack-v0"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


def main():

    # env = gym.make("xgym/stack-v0")
    env = Human(mode="manual")
    env.reset()

    for ep in range(1):

        all_imgs = []

        for i in tqdm(range(int(1e3)), desc=f"Collecting episode {ep}"):
            imgs = env.look()
            all_imgs.append(imgs)
            # env.render(refresh=True)
            time.sleep(0.1)

        # make them into a video and save to disk
        for k, v in all_imgs[0].items():
            frames = [x[k] for x in all_imgs]
            with imageio.get_writer(f"ep{ep}_{k}.mp4", fps=10) as writer:
                for frame in frames:
                    writer.append_data(frame)

    env.close()
    quit()


def sandbox():
    # env.go(RS(cartesian=[75, -125, 125], aa=[0, 0, 0]), relative=True)
    # [3.133083, 0.013429, -1.608588]

    # rpy=[np.pi, 0, -np.pi/2]
    # print(env.angles)
    # quit()
    # env.go(RS(cartesian=env.cartesian, aa=rpy), relative=False)

    # print(env.rpy)
    # quit()
    # rpy = np.array(env.rpy) + np.array([0.0,0,-0.1])
    # rpy[1] = 0
    # env.go(RS(cartesian=env.cartesian, aa=rpy), relative=False)

    # action = [0.1, 0.1, 0.1, 0, 0, 0, 0]

    pass


if __name__ == "__main__":
    main()
