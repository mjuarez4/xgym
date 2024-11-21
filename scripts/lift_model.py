import os
import os.path as osp
import time
from dataclasses import dataclass, field

import cv2
import draccus
import envlogger
import gymnasium as gym
import jax
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
from xgym.utils import camera as cu
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

    #model = ModelController("carina.cs.luc.edu", 8001, ensemble=True)
    model = ModelController("aisec-102.cs.luc.edu", 8001, ensemble=True)
    #model = ModelController("dijkstra.cs.luc.edu", 8001, ensemble=True)
    model.reset()

    # env = gym.make("xgym/stack-v0")
    _env = Lift(out_dir=cfg.data_dir, random=False)

    # ds = tfds.load("xgym_lift_single", split="train")

    freq = 5  # hz
    dt = 1 / freq

    with _env as env:

        for ep in tqdm(range(10), desc="Episodes"):
            obs = env.reset()
            env.set_mode(7)
            time.sleep(0.4)
            env.start_record()

            # timestep = env.reset()
            # obs = timestep.observation
            for _ in tqdm(range(55), desc=f"EP{ep}"):  # 3 episodes

                tic = time.time()
                print("\n" * 3)

                obs["img"] = jax.tree_map(
                    lambda x: cv2.resize(np.array(x), (224, 224)), obs["img"]
                )

                print(obs["img"].keys())

                myimg = obs["img"]["camera_6"]
                primary = obs["img"]["camera_10"]  #switched these (cam10->cam6, cam6->cam10) for 133026 eval

                cv2.imshow(
                    "data Environment",
                    cv2.cvtColor(cu.tile(cu.writekeys(obs["img"])), cv2.COLOR_RGB2BGR),
                )
                cv2.waitKey(1)  # 1 ms delay to allow for rendering

                # action = model(obs["img"]["camera_0"], obs['img']['wrist']).copy()
                actions = model(
                    primary=primary, high=myimg, wrist=obs["img"]["wrist"]
                ).copy()
                # action = action[0] # take first of 4 steps in chunk

                if not model.ensemble:
                    for a, action in enumerate(actions):
                        tic = time.time() if a else tic
                        action[:3] *= int(1e3)
                        print(action.shape)
                        action[3:6] = 0
                        action[-1] = 0.2 if action[-1] < 0.8 else 1  # less gripper
                        # action = action / 2
                        print(f"action: {[round(x,4) for x in action.tolist()]}")
                        # _env.render(mode="human")

                        obs, reward, done, info = env.step(action)
                        toc = time.time()
                        elapsed = toc - tic
                        time.sleep(max(0, dt - elapsed))  # 5hz
                        print("ensembling") 
                else:
                      
                    action = actions
                    action[3:6] = 0
                    action[:3] *= int(1e3)
                    action[-1] = 0.2 if action[-1] < 0.8 else 1  # less gripper
                    print(f"action: {[round(x,2) for x in action.tolist()]}")
                    obs, reward, done, info = env.step(action)
                    toc = time.time()
                    elapsed = toc - tic
                    time.sleep(max(0, dt - elapsed))  # 5hz

                print(f"done: {done}")
                if done:
                    break

                # timestep = env.step(action)
                # obs = timestep.observation

            env.stop_record()
            env.flush()
            #env.auto_reset()

    env.close()
    _env.close()

    quit()


if __name__ == "__main__":
    main()
