import enum
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import cv2
import draccus
import jax
import jax.numpy as jnp
import lerobot
import numpy as np
import torch
import tyro
from flax.traverse_util import flatten_dict
from jax import numpy as jnp
from lerobot.common.datasets.lerobot_dataset import \
    HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rich.pretty import pprint
from tqdm import tqdm

import xgym
from xgym.lrbt.convert import (DEFAULT_DATASET_CONFIG, FRAME_MODE, MOTORS,
                               DatasetConfig, Embodiment, Task, create)
from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose
from xgym.rlds.util.trajectory import binarize_gripper_actions, scan_noop
from xgym.viz.mano import overlay_palm, overlay_pose


np.set_printoptions(suppress=True, precision=3)

@dataclass
class Config:
    dir: str  # the dir of the memmap files
    branch: str  # the branch to push to ie: "v0.1"
    repo_id: str  # the repo id to push to ie: "mhyatt000/demo"


def main(cfg: Config):

    #
    # this is the main part that needs to change
    #

    root = Path(cfg.dir)
    ds = list(root.rglob("*.dat"))

    spec = lambda tree: jax.tree.map(lambda x: x.shape, tree)
    take = lambda tree, _i: jax.tree.map(lambda x: x[_i], tree)

    times = []

    # task = "embodiment:Human, task:pick up the red block"  # hardcoded for now
    taskfile = next(Path(cfg.dir).glob("*.npy"))
    _task = taskfile.stem.replace("_", " ")
    _lang = np.load(taskfile)

    dataset = None
    for f in tqdm(ds, total=len(ds)):

        try:
            info, ep = xgym.viz.memmap.read(f)
            cams = [k for k in info["schema"].keys() if "camera" in k]
            if len(cams) < 2:
                raise ValueError(f"Not enough cameras {cams}")
        except Exception as e:
            xgym.logger.error(f"Error reading {f}")
            xgym.logger.error(e)
            continue

        if len(ep["time"]) < 200:  # 4 seconds @ 50hz
            xgym.logger.error(f"Episode too short {len(ep['time'])}")
            continue
        pprint(spec(ep))

        leader = ep['gello_joints']
        leader_grip = leader[:,-1:]
        # pprint(leader_grip)

        n = len(ep["time"])
        steps = {
            "observation": {
                "image": {k.split("/")[-1]: ep[k] for k in ep if "camera" in k},
                "proprio": {k.split("_")[-1]: ep[k] for k in ep if "xarm" in k},
            },
        }
        steps['observation']['proprio']['gripper'] = leader_grip
        pprint(spec(steps))

        # steps = [x for x in ep["steps"]]
        # steps = jax.tree.map(lambda x: np.array(x), steps)
        # steps = jax.tree.map(lambda *_x: np.stack(_x, axis=0), *steps)

        for k in [
            "discount",
            "reward",
            "is_first",
            "is_last",
            "is_terminal",
            "action",
        ]:
            if k in steps:
                steps.pop(k)

        obs = steps.pop("observation")
        img = obs.pop("image")

        for k in ['overhead','high']:
            if k in img:
                img.pop(k)
                # img["high"] = img.pop("overhead")

        # img["low"] = img.pop("worm") # already named to low
        # to encode as mp4 with ffmpeg
        img = jax.tree.map(lambda x: x / 255, img)
        obs["image"] = img

        # force to be joint actions
        state = obs.pop("proprio")
        # state['gripper'] /= 850 # no because we use leader grip
        state['position'] = state.pop('pose') / 1e3
        obs["state"] = state


        # steps["action"] = jax.tree.map(
                # lambda x: np.concatenate([x[:, 1:], x[:, -1:]], axis=-1), state
                # )
        # pprint(spec(steps["action"]))

        steps["observation"] = obs

        lang = {
            "lang": np.array([_lang for _ in range(n)]),
            "task": _task,
        }
        steps["lang"] = lang["lang"]

        # where state gripper is nan
        mask = ~np.isnan(state["gripper"]).reshape(-1)
        steps = jax.tree.map(select := lambda x: x[mask], steps)
        n = mask.sum()

        print(f"Kept {mask.sum()} of {len(mask)} steps")

        # thresh = 1e-3
        # noops = np.array(scan_noop(jnp.array(pos), threshold=thresh))
        # mask = ~noops
        # filter noop joints
        # jpos = np.concatenate([state["joints"], state["gripper"]], axis=-1)
        # jnoop = np.array(scan_noop(jnp.array(jpos), threshold=thresh, binary=False))
        # jmask = ~jnoop
        # mask = np.logical_and(mask, jmask)
        # mask = jmask

        # print(f"Kept {mask.sum()} of {n} steps")
        # steps = jax.tree.map(select := lambda x: x[mask], steps)

        # pprint(spec(steps))
        step0 = take(steps, 0)

        pprint(spec(step0))
        # quit()

        if dataset is None:
            pprint(spec(step0))
            dataset = create(
                repo_id=cfg.repo_id,
                robot_type="xarm",
                example=deepcopy(step0),
            )

        # pprint(jax.tree.map(lambda x: (x.shape,x.dtype), steps))
        task = str(lang["task"][0])
        for i in tqdm(range(n), leave=False):
            step = take(steps, i)
            step = flatten_dict(step, sep=".")
            step = jax.tree.map(lambda x: torch.from_numpy(x).float(), step)
            dataset.add_frame(step | {"task": task})
        dataset.save_episode()

    tags = []
    dataset.push_to_hub(
        branch=cfg.branch,
        tags=tags,
        private=False,
        push_videos=True,
    )
    # for i, s in tqdm(enumerate(steps), total=len(steps), leave=False):


if __name__ == "__main__":
    main(tyro.cli(Config))
