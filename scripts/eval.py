import os
import os.path as osp
import time
from dataclasses import dataclass, field

import cv2
import draccus
import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pynput import keyboard
from tqdm import tqdm

from xgym.controllers import KeyboardController, ScriptedController
from xgym.gyms import Base, Lift, Stack
from xgym.model_controllers import ModelController
from xgym.utils import boundary as bd
from xgym.utils import camera as cu
from xgym.utils.boundary import PartialRobotState as RS


import draccus

@dataclass
class RunCFG:
    host: str
    port: int

    base_dir: str = osp.expanduser("~/data")
    time: str = time.strftime("%Y%m%d-%H%M%S")
    task: str = ""
    ensemble: bool = True
    nsteps: int = 100
    
    def __post_init__(self):
        self.check()

        self.env_name: str = f"xgym-eval-{self.task}-{self.time}"
        self.data_dir: str = osp.join(self.base_dir, self.env_name)

    def check(self):
        assert self.task, "Please specify a task"


@draccus.wrap()
def main(cfg: RunCFG):

    print(cfg)

    os.makedirs(cfg.data_dir, exist_ok=True)

    model = ModelController(
	cfg.host,
	cfg.port,
        ensemble=cfg.ensemble,
        task=cfg.task,
    )

    model.reset()

    # env = gym.make("xgym/stack-v0")
    env = Lift(out_dir=cfg.data_dir, random=False)
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
        for _ in tqdm(range(cfg.nsteps), desc=f"EP{ep}"):  # 3 episodes

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
            proprio = np.array(proprio).astype(np.float64)

            # TODO the datasets should be fixed to regular proprio vals in mm and 0-1
            proprio[:3] = proprio[:3] / 1e3
            proprio[-1] = proprio[-1] / env.GRIPPER_MAX
            actions = model(
                primary=obs["img"]["worm"],
                high=obs["img"]["overhead"],
                wrist=obs["img"]["wrist"],
                side=obs["img"]["side"],
                # proprio=proprio,
            ).copy()

            print(actions.round(3))
            for a, action in enumerate(actions):

                action[:3] *= int(1e3)
                print(action.shape)
                # action[3:6] = action[3:6].clip(-0.25,0.25)
                # input("Press Enter to continue...")
                action[-1] *= 0.8 if action[-1] < 0.5 else 1
                # action[-1] *= 0 if action[-1] < 0.2 else 1
                # action[-1] = 1 if action[-1] > 0.8 else action[-1]

                print(f"action: {[round(x,4) for x in action.tolist()]}")

                obs, reward, done, info = env.step(action)
                # time.sleep(dt)

                toc = time.time()
                elapsed = toc - tic
                time.sleep(max(0, dt - elapsed))  # 5hz
                tic = time.time() if a else tic

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
