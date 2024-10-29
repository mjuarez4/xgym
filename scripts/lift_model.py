import os
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
from xgym.gyms import Base, Lift, Stack
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS


@dataclass
class RunCFG:

    base_dir: str = osp.expanduser("~/data")
    time: str = time.strftime("%Y%m%d-%H%M%S")
    env_name: str = f"xgym-liftmodel-v0-{time}"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


def main():

    os.makedirs(cfg.data_dir, exist_ok=True)
    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name=cfg.env_name,
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
        action_info=tfds.features.Tensor(shape=(7,), dtype=np.float64),
        reward_info=tfds.features.Tensor(shape=(), dtype=np.float64),
        discount_info=tfds.features.Tensor(shape=(), dtype=np.float64),
    )

    # env: Base = gym.make("luc-base")

    model = ModelController("carina.cs.luc.edu", 8001)
    model.reset()

    # env = gym.make("xgym/stack-v0")
    _env = Lift()

    with envlogger.EnvLogger(
        DMEnvFromGym(_env),
        backend=TFDSWriter(
            data_directory=cfg.data_dir,
            split_name="train",
            max_episodes_per_file=50,
            ds_config=dataset_config,
        ),
    ) as env:

    # with _env as env:

        for ep in tqdm(range(10), desc="Episodes"):
            # obs = env.reset()
            timestep = env.reset()
            obs = timestep.observation
            for _ in tqdm(range(15), desc=f"EP{ep}"):  # 3 episodes

                print("\n" * 3)
                # action = model(obs["img"]["camera_0"], obs['img']['wrist']).copy()
                action = model(obs["img"]["camera_0"]).copy()
                action[3:6] = 0
                # action[-1] *= 0.8 if action[-1] < 0.8 else 1 # less gripper
                # action = action / 2
                print(f"action")
                print(action.tolist())
                _env.render(mode="human")
                # obs, reward, done, info = env.step(action)
                timestep = env.step(action)
                obs = timestep.observation

    env.close()
    _env.close()

    quit()


if __name__ == "__main__":
    main()
