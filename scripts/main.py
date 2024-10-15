import os
from tqdm import tqdm
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

from xgym.controllers import (KeyboardController, ModelController,
                              ScriptedController)
from xgym.gyms import Base
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS


@dataclass
class RunCFG:

    base_dir: str = osp.expanduser("~/data")
    env_name: str = "luc-base"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


def main():

    # env: Base = gym.make("luc-base")
    controller = KeyboardController()
    controller.register(keyboard.Key.space, lambda: env.stop(toggle=True))
    _env = Base()

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
                        "wrist": tfds.features.Tensor(
                            shape=(640, 640, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        ),
        action_info=tfds.features.Tensor(shape=(7,), dtype=np.float32),
        reward_info=np.float64,
        discount_info=np.float64,
    )

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
    env = _env

    for _ in tqdm(range(3)): # 3 episodes

        obs = env.reset()
        print(f"entering loop")
        for i in tqdm(range(5),leave=False):
            time.sleep(0.01)

            # action = controller(obs)
            mode = "all"
            if mode == "cart":
                action = np.random.normal(0, 1, 7) * 25
                action[3:6] = 0
            elif mode == "rot":
                action = np.random.normal(0, 1, 7) * 0.05
                action[:3] = 0
            else:
                action = np.random.normal(0, 1, 7) * 25
                action[3:6] = np.random.normal(0, 1, 3) * 0.05  # rotation

            # gripper is x100 but only active 20% of the time
            gripper = np.random.choice([0, 1], p=[0.6, 0.4])
            action[6] *= gripper * 4
            action = action.astype(np.float32)

            """
            action = controller(obs)
            while action.sum() == 0:
                action = controller(obs)
            """

            print()
            print(action)

            # things = env.step()
            # obs, reward, truncated, terminated, info = env.step(env.action_space.sample())
            things = env.step(action)
            # print(things)
            _env.render()
            print("one step")

    env.close()
    controller.close()
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
