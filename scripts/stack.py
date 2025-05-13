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
from envlogger.backends.tfds_backend_writer import TFDSBackendWriter as TFDSWriter
from envlogger.testing import catch_env
from pynput import keyboard
from tqdm import tqdm

from xgym.controllers import KeyboardController, ModelController, ScriptedController
from xgym.gyms import Base, Stack
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS


@dataclass
class RunCFG:

    base_dir: str = osp.expanduser("~/data")
    env_name: str = "xgym-stack-v0"
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
    _env = Stack(mode="manual")

    with envlogger.EnvLogger(
        DMEnvFromGym(_env),
        backend=TFDSWriter(
            data_directory=cfg.data_dir,
            split_name="train",
            max_episodes_per_file=50,
            ds_config=dataset_config,
        ),
    ) as env:

        for _ in tqdm(range(50)):  # 3 episodes

            env.reset()

            ### planner algorithm
            p1, p2 = np.array(_env.p1.joints), np.array(_env.p2.joints)
            ready = np.array(_env.ready.joints)

            # go to p1 in 3 steps
            steps = [(p1 * a) + (ready * (1 - a)) for a in np.linspace(0, 1, 5)]
            for s in steps:
                print()
                print(s)
                absolute = _env.kin_fwd(s) + [0]
                act = np.array(absolute) - _env.position.to_vector()
                act[-1] = 0.00
                print(act)
                env.step(act.astype(np.float64))

            env.step(
                np.array([0, 0, 0, 0, 0, 0, 280 - _env.gripper]).astype(np.float64)
            )
            env.step(np.array([0, 0, 100, 0, 0, 0, 0]).astype(np.float64))

            p2 = RS.from_vector(_env.kin_fwd(p2))
            p2.cartesian[2] += 50
            p2.gripper = 0
            p2 = np.array(_env.kin_inv(p2.to_vector()[:-1]))
            pos = _env.position.joints
            steps = [(p2 * a) + (pos * (1 - a)) for a in np.linspace(0, 1, 5)]

            for s in steps:
                absolute = _env.kin_fwd(s) + [0]
                act = np.array(absolute) - _env.position.to_vector()
                act[-1] = 0
                print(act)
                env.step(act.astype(np.float64))

            env.step(np.array([0, 0, 0, 0, 0, 0, 50]).astype(np.float64))

    _env.close()

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
