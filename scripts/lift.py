import os
import os.path as osp
import random
import time
from dataclasses import dataclass, field

import cv2
import draccus
import envlogger
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from bsuite.utils.gym_wrapper import DMEnvFromGym, GymFromDMEnv
from envlogger.backends.tfds_backend_writer import TFDSBackendWriter as TFDSWriter
from envlogger.testing import catch_env
from pynput import keyboard
from tqdm import tqdm

from xgym.controllers import KeyboardController, ModelController, ScriptedController
from xgym.gyms import Base, Lift, Stack
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS


@dataclass
class RunCFG:

    base_dir: str = osp.expanduser("~/data")
    time: str = time.strftime("%Y%m%d-%H%M%S")
    env_name: str = f"xgym-lift-v0-{time}"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


def main():

    os.makedirs(cfg.data_dir, exist_ok=True)
    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name="luc-base",
        observation_info=tfds.features.FeaturesDict(
            {
                "robot": tfds.features.FeaturesDict(
                    {
                        "joints": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                        "position": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                    }
                ),
                "img": tfds.features.FeaturesDict(
                    {
                        "camera_0": tfds.features.Tensor(
                            shape=(640, 640, 3), dtype=np.uint8
                        ),
                        "camera_1": tfds.features.Tensor(
                            shape=(640, 640, 3), dtype=np.uint8
                        ),
                        "wrist": tfds.features.Tensor(
                            shape=(640, 640, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        ),
        action_info=tfds.features.Tensor(shape=(7,), dtype=np.float64),
        reward_info=tfds.features.Tensor(shape=(), dtype=np.float64),
        discount_info=tfds.features.Tensor(shape=(), dtype=np.float64),
    )

    # env: Base = gym.make("luc-base")

    # env = gym.make("xgym/stack-v0")
    _env = Lift(manual=True, out_dir=cfg.data_dir)

    """
    while True:
        imgs = _env.look()
        imgs = np.concatenate(list(imgs.values()), axis=1)
        cv2.imshow("image", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(50)
    """

    """
    with envlogger.EnvLogger(
        DMEnvFromGym(_env),
        backend=TFDSWriter(
            data_directory=cfg.data_dir,
            split_name="train",
            max_episodes_per_file=50,
            ds_config=dataset_config,
        ),
    ) as env:
    """
    with _env as env:

        for _ in tqdm(range(100), desc="episodes"):

            env.reset()
            time.sleep(0.2)
            env.start_record()

            ### planner algorithm
            p1 = _env.p1
            ready = _env.position

            p1.aa = bd.minimize(p1.aa)
            ready.aa = bd.minimize(ready.aa)

            action = p1.to_vector() - ready.to_vector()
            action[3:6] = 0
            action[-1] = 1
            env.step(action)

            CUBE_W = 280
            grip = CUBE_W / _env.GRIPPER_MAX
            # env.step(np.array([0, 0, 0, 0, 0, 0, grip]).astype(np.float64))
            env.step(np.array([0, 0, 100, 0, 0, 0, grip]).astype(np.float64))

            env.stop_record()
            env.flush()
            _env.auto_reset()

    # super(_env).reset()
    try:
        env.close()
        _env.close()
    except Exception as e:
        raise e


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
