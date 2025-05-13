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

from xgym.controllers import KeyboardController, ModelController, ScriptedController
from xgym.gyms import Base, Lift, Stack
from xgym.utils import boundary as bd
from xgym.utils import camera as cu
from xgym.utils.boundary import PartialRobotState as RS

np.set_printoptions(suppress=True)


@dataclass
class RunCFG:

    base_dir: str = osp.expanduser("~/data")
    time: str = time.strftime("%Y%m%d-%H%M%S")
    env_name: str = f"xgym-liftmodel-v0-{time}"
    data_dir: str = osp.join(base_dir, env_name)


cfg = RunCFG()


def new_target(target, action):
    target = target.copy()
    target += action
    target[-1] = action[-1]
    return target


def action_from_target(target, env):
    clean = lambda x: np.mod(np.abs(x), 2 * np.pi) * np.sign(x)
    pos = env.position.to_vector()
    pos[3:6] = clean(pos[3:6])
    action = target - pos
    action[-1] = target[-1]

    # action[3:6] =

    return action


def make_target(env):
    clean = lambda x: np.mod(np.abs(x), 2 * np.pi) * np.sign(x)
    target = env.position.to_vector()
    target[3:6] = clean(target[3:6])
    target[-1] = 0.95
    return target


def main():

    os.makedirs(cfg.data_dir, exist_ok=True)

    env = Lift(out_dir=cfg.data_dir, random=False)

    freq = 5  # hz
    dt = 1 / freq

    model = ModelController("carina.cs.luc.edu", 8001, ensemble=False)
    model.reset()

    ds = tfds.load("xgym_lift_single", split="train")
    # ds = tfds.load("xgym_play_single", split="train")

    target = None

    for ep in tqdm(ds, desc="Episodes"):

        obs = env.reset()
        env.set_mode(7)
        time.sleep(0.4)
        env.start_record()

        # timestep = env.reset()
        # obs = timestep.observation
        _action = None
        for step in tqdm(ep["steps"], desc=f"EP{ep}"):  # 3 episodes

            step = jax.tree_map(lambda x: np.array(x), step)
            spec = lambda x: x.shape
            print(jax.tree.map(spec, step))

            position = step["observation"]["proprio"]["position"]
            print(f"position: {position}")

            _pos = env.position.to_vector()
            if _action is None:  # get to the random starting point
                _action = position - _pos
                env.logger.error(f"stepping to the start")
                # env.step(_action)
                # time.sleep(2)
            else:
                _action = position - _pos
            _action[-1] = position[-1] / env.GRIPPER_MAX

            tic = time.time()
            print("\n" * 3)

            images = jax.tree_map(
                lambda x: cv2.resize(np.array(x), (224, 224)),
                step["observation"]["image"],
            )
            print(images.keys())

            cv2.imshow(
                "data Environment",
                cv2.cvtColor(cu.tile(cu.writekeys(images)), cv2.COLOR_RGB2BGR),
            )
            cv2.waitKey(1)  # 1 ms delay to allow for rendering

            obs["img"] = jax.tree_map(
                lambda x: cv2.resize(np.array(x), (224, 224)), obs["img"]
            )
            cv2.imshow(
                "irl Environment",
                cv2.cvtColor(cu.tile(cu.writekeys(obs["img"])), cv2.COLOR_RGB2BGR),
            )
            cv2.waitKey(1)  # 1 ms delay to allow for rendering

            """ inference from dataset
            action = model(
                primary=images["window"],
                high=images["overhead"],
                wrist=images["wrist"],
            ).copy()
            """

            actions = model(
                primary=obs["img"]["worm"],
                high=obs["img"]["overhead"],
                wrist=obs["img"]["wrist"],
                side=obs["img"]["side"],
                # proprio=proprio.tolist(),
            ).copy()

            for action in actions:

                action[:3] *= int(1e3)
                action[-1] *= 0.95 if action[-1] < 0.7 else 1
                # action = action / 2
                print(f"action: {[round(x,4) for x in action.tolist()]}")
                # _env.render(mode="human")

                # target = make_target(env) if target is None else target
                # target = new_target(target, action)
                # action = action_from_target(target, env)
                # print("target", target.round(3))
                print(action.round(3))

                obs, reward, done, info = env.step(action)
                time.sleep(3 * dt)
                # toc = time.time()
                # elapsed = toc - tic
                # time.sleep(max(0, dt - elapsed))  # 5hz
                # tic = time.time() if a else tic

                print(f"done: {done}")
                if done:
                    break

            if done:
                break

        env.stop_record()
        env.flush()
        # env.auto_reset()

    env.close()
    quit()


if __name__ == "__main__":
    main()
