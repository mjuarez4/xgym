import os
import os.path as osp
import time
from dataclasses import dataclass, field

import cv2
import draccus
# import envlogger
import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from bsuite.utils.gym_wrapper import DMEnvFromGym, GymFromDMEnv
# from envlogger.backends.tfds_backend_writer import \
# TFDSBackendWriter as TFDSWriter
# from envlogger.testing import catch_env
from pynput import keyboard
from tqdm import tqdm

from xgym.controllers import KeyboardController, ScriptedController
from xgym.gyms import Base, Lift, Stack
from xgym.model_controllers import ModelController
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

    # @ethan TODO make this a configurable parameter? see scripts/reader_rlds and github.com/dlwh/draccus
    model = ModelController(
        "carina.cs.luc.edu",
        8001,
        ensemble=False,
        task="stack",
    )
    # model = ModelController("aisec-102.cs.luc.edu", 8001, ensemble=True)
    # model = ModelController("dijkstra.cs.luc.edu", 8001, ensemble=True)
    model.reset()

    # env = gym.make("xgym/stack-v0")
    env = Lift(out_dir=cfg.data_dir, random=True)
    env.logger.warning(model.tasks[model.task])

    # ds = tfds.load("xgym_lift_single", split="train")

    freq = 5  # hz
    dt = 1 / freq

    for ep in tqdm(range(100), desc="Episodes"):
        obs = env.reset()
        env.set_mode(7)
        time.sleep(0.4)
        env.start_record()

        # timestep = env.reset()
        # obs = timestep.observation
        for _ in tqdm(range(75), desc=f"EP{ep}"):  # 3 episodes

            tic = time.time()
            print("\n" * 3)

            # remap obs
            print(obs["img"].keys())

            obs["img"] = jax.tree_map(
                lambda x: cv2.resize(np.array(x), (224, 224)), obs["img"]
            )

            cv2.imshow(
                "data Environment",
                cv2.cvtColor(cu.tile(cu.writekeys(obs["img"])), cv2.COLOR_RGB2BGR),
            )
            cv2.waitKey(1)  # 1 ms delay to allow for rendering

            proprio = obs["robot"]["position"]
            proprio = np.array(proprio)
            proprio[:3] = proprio[:3] / 1e3
            proprio[-1] = proprio[-1] / env.GRIPPER_MAX

            actions = model(
                primary=obs["img"]["worm"],
                high=obs["img"]["overhead"],
                wrist=obs["img"]["wrist"],
                side=obs["img"]["side"],
                # proprio=proprio.tolist(),
            ).copy()

            print(actions.round(3))
            for a, action in enumerate(actions):

                action[:3] *= int(1e3)
                print(action.shape)
                action[-1] *= 0.8 if action[-1] < 0.5 else 1
                # action[-1] *= 0 if action[-1] < 0.2 else 1
                # action[-1] = 1 if action[-1] > 0.8 else action[-1]

                print(f"action: {[round(x,4) for x in action.tolist()]}")

                obs, reward, done, info = env.step(action)
                time.sleep(dt)

                # toc = time.time()
                # elapsed = toc - tic
                # time.sleep(max(0, dt - elapsed))  # 5hz
                # tic = time.time() if a else tic

                print(f"done: {done}")
                if done:
                    break
            if done:
                break

            # timestep = env.step(action)
            # obs = timestep.observation

        env.stop_record()
        env.flush()
        # env.auto_reset()

    env.close()
    quit()


if __name__ == "__main__":
    main()
