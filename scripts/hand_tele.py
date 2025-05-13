import os
import os.path as osp
import time
from dataclasses import dataclass, field

import cv2
import draccus
import envlogger
import gymnasium as gym
import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from bsuite.utils.gym_wrapper import DMEnvFromGym, GymFromDMEnv
from envlogger.backends.tfds_backend_writer import TFDSBackendWriter as TFDSWriter
from envlogger.testing import catch_env
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pynput import keyboard
from tqdm import tqdm

from xgym.controllers import (
    HamerController,
    KeyboardController,
    ModelController,
    ScriptedController,
    SpaceMouseController,
)
from xgym.gyms import Base, Lift, Stack
from xgym.utils import boundary as bd
from xgym.utils import camera as cu
from xgym.utils.boundary import PartialRobotState as RS


@dataclass
class RunCFG:

    task: str = input("Task: ").lower()
    base_dir: str = osp.expanduser("~/data")
    time: str = time.strftime("%Y%m%d-%H%M%S")
    env_name: str = f"xgym-sandbox-{task}-v0-{time}"
    data_dir: str = osp.join(base_dir, env_name)
    nsteps: int = 30


plt.switch_backend("Agg")


def plot(actions):
    x, y, z, roll, pitch, yaw, gripper = actions.T
    time = np.arange(len(x))

    fig, ax = plt.subplots()
    ax.plot(time, x, label="x")
    ax.plot(time, y, label="y")
    ax.plot(time, z, label="z")
    ax.plot(time, roll, label="roll")
    ax.plot(time, pitch, label="pitch")
    ax.plot(time, yaw, label="yaw")
    ax.plot(time, gripper, label="gripper")
    ax.legend()

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


@draccus.wrap()
def main(cfg: RunCFG):

    os.makedirs(cfg.data_dir, exist_ok=True)

    agent = HamerController("aisec-102.cs.luc.edu", 8001)

    # env = gym.make("xgym/stack-v0")
    env = Lift(out_dir=cfg.data_dir, random=False)

    # ds = tfds.load("xgym_lift_single", split="train")

    freq = 10  # hz
    dt = 1 / freq

    hist = np.zeros(7)

    for ep in tqdm(range(100), desc="Episodes"):

        obs = env.reset()
        print(obs.keys())

        env.set_mode(7)
        time.sleep(0.4)
        env.start_record()

        # timestep = env.reset()
        # obs = timestep.observation

        delta = np.zeros(7)
        k0 = np.zeros((21, 3))
        fingermin = 1e3
        fingermax = 1e-8

        for _ in tqdm(range(int(cfg.nsteps) * freq), desc=f"EP{ep}"):  # 3 episodes

            tic = time.time()
            print("\n" * 3)

            # obs["img"] = jax.tree_map(
            # lambda x: cv2.resize(np.array(x), (224, 224)), obs["img"]
            # )

            # print(obs["img"].keys())

            # myimg = obs["img"]["camera_10"]
            # primary = obs["img"]["camera_6"]

            np.set_printoptions(suppress=True)

            hands = agent(cv2.resize(obs["img"]["worm"], (224, 224)))
            print(hands.keys())

            if len(hands.keys()):

                kp = hands["pred_keypoints_3d"]
                kp = kp[0] if len(kp.shape) == 3 else kp

                camtfull = hands["pred_cam_t_full"]
                camtfull = camtfull[0] if len(camtfull.shape) == 2 else camtfull
                kp = kp + camtfull

                delta = kp - k0 if k0.any() else k0
                k0 = kp

                delta = delta.mean(0)  # was palm 0 now mean

                print(kp)
                print(kp.shape)

                fingers = np.array([4, 8, 12, 20])

                # for a,b in fingers, measure the distance
                # sum all the distances
                import itertools

                combos = itertools.product(fingers, fingers)
                dist = np.sum([np.linalg.norm(kp[a] - kp[b]) for a, b in combos])

                fingermin = min(fingermin, dist)
                fingermax = max(fingermax, dist)
                grip = (dist - fingermin) / (fingermax - fingermin + 1e-8)

                # z is depth for camera... y for base
                # y is height for camera... z for base
                delta = np.array(
                    [
                        -15 * delta[2],
                        100 * delta[0],
                        -100 * delta[1],
                        0,
                        0,
                        0,
                        grip,
                    ]
                )

            # print(type(hands))
            # if isinstance(hands,dict):
            # print(hands.keys())

            action = delta

            # action[-1] += env.gripper / env.GRIPPER_MAX
            print(f"action: {action.round(4)}")

            pose = env.position.to_vector()
            pose[:3] /= int(1e3)
            pose[-1] /= env.GRIPPER_MAX
            # hist = np.vstack([hist, pose])
            # img = plot(hist)

            # action[:3] *= int(1e2)
            # action[-1] =  0.2 if action[-1] < 0.8 else 1  # less gripper

            cv2.imshow(
                "data Environment",
                cv2.cvtColor(cu.tile(cu.writekeys(obs["img"])), cv2.COLOR_RGB2BGR),
            )
            cv2.waitKey(1)  # 1 ms delay to allow for rendering

            env.send(action)

            done = env._done

            toc = time.time()
            elapsed = toc - tic
            # time.sleep(max(0, dt - elapsed))  # 5hz

            print(f"done: {done}")
            if done:
                break

            obs, done, info = env.observation(), False, {}

        env.stop_record()
        env.flush()
        # env.auto_reset()

    env.reset()
    env.close()

    quit()


if __name__ == "__main__":
    main()
