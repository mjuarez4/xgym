import enum
import shutil
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
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
# from mano_pipe_v3 import remap_keys, select_keys
from rich.pretty import pprint
from tqdm import tqdm
from webpolicy.deploy.client import WebsocketClientPolicy

import xgym
from xgym import BASE
from xgym.calibrate.april import Calibrator
from xgym.lrbt.convert import (DEFAULT_DATASET_CONFIG, FRAME_MODE, MOTORS,
                               DatasetConfig, Embodiment, Task, create)
from xgym.lrbt.from_memmap import Config
from xgym.lrbt.util import get_taskinfo
from xgym.rlds.util import (add_col, apply_persp, apply_uv, apply_xyz,
                            perspective_projection, remove_col, solve_uv2xyz)
from xgym.rlds.util.render import render_openpose
from xgym.rlds.util.trajectory import binarize_gripper_actions, scan_noop
from xgym.viz.mano import overlay_palm, overlay_pose
import threading
from typing import Union

np.set_printoptions(suppress=True, precision=3)


import logging
logger = logging.getLogger(__name__)


def spec(thing: dict[str, np.ndarray]):
    """Returns the shape of each key in the dict."""
    return jax.tree.map(lambda x: x.shape, thing)


# 4x4 matx
T_xflip = np.array(
    [
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


@dataclass
class HamerConfig:
    host: str
    port: int


def remap_keys(out: dict):
    out = {k.replace("pred_", "").replace("_params", ""): v for k, v in out.items()}
    out = {k.replace("keypoints_", "kp"): v for k, v in out.items()}
    return out


def select_keys(out: dict):
    """Selects keys to keep in the output.
    as a side effect, prepares the keypoints_3d for further processing
    TODO break into separate func?
    """

    out["kp3d"] += out.pop("cam_t_full")[:, None, :]
    keep = [
        # "box_center",
        # "box_size",
        "img",
        "img_wrist",
        # "img_size",
        # "personid",
        # "cam",
        # "cam_t",
        # "cam_t_full", # used above
        "kp2d",  # these are wrt the box not full img ... solve for 2d
        "kp3d",
        "mano.betas",
        "mano.global_orient",
        "mano.hand_pose",
        # "vertices",
        "right",
        # "focal_length", # i think we use the scaled one instead
        "scaled_focal_length",
    ]
    return {k: out[k] for k in keep if k in out}


def check_shapes(out: dict):
    """sometimes hamer returns 2 predictions"""
    # TODO in the future , filter by mIOU on bboxes

    for k in [ 'kp3d', 'mano.betas', 'mano.global_orient', 'mano.hand_pose', 'right' , 'img_wrist']:
        if out[k].shape and out[k].shape[0] == 2: # some have no shape 
            logger.warning(f"Warning: {k} has 2 predictions, taking first")
            out[k] = out[k][0]
    return out


def postprocess(out: dict, frame: np.ndarray):

    out = jax.tree.map(lambda x: x.copy(), out)
    out = remap_keys(out)
    box = {
        "center": out["box_center"][0],
        "size": out["box_size"][0],
        "kp2d": out.pop("kp2d")[0],  # only relevant to box
    }

    right = bool(out["right"][0])
    left = not right

    if left:
        n = len(out["kp3d"])
        kp3d = add_col(out["kp3d"])
        kp3d = remove_col((kp3d[0] @ T_xflip)[None])
        out["kp3d"] = np.concatenate([kp3d for _ in range(n)])

    out = select_keys(out)

    f = out["scaled_focal_length"]
    P = perspective_projection(f, H=frame.shape[0], W=frame.shape[1])
    points2d = apply_persp(out["kp3d"], P)[0, :, :-1]
    out["kp2d"] = points2d

    out = out | {"box": box}
    squeeze = lambda arr: jax.tree.map(lambda x: x.squeeze(), arr)
    out = squeeze(out)

    def maybe_unsqueeze(x):
        return x.reshape((-1)) if x.ndim <= 1 else x

    # is_leaf = lambda x: isinstance(x, (np.array, tuple, list))
    out = jax.tree.map(lambda x: np.array(x), out)
    out = check_shapes(out)
    out = jax.tree.map(maybe_unsqueeze, out)
    out = jax.tree.map(lambda x: x.astype(np.float32), out)

    return out


@dataclass
class MyConfig(Config):
    hamer: HamerConfig = field(default_factory=HamerConfig)


def main(cfg: MyConfig):

    root = Path(cfg.dir)
    ds = list(root.rglob("ep*.npz"))

    take = lambda tree, _i: jax.tree.map(lambda x: x[_i], tree)

    times = []

    taskinfo = get_taskinfo(cfg.dir)
    _task, _lang = taskinfo["task"], taskinfo["lang"]

    pprint(cfg)

    dataset = None
    for f in tqdm(ds, total=len(ds)):

        client = WebsocketClientPolicy(host=cfg.hamer.host, port=cfg.hamer.port)
        ep = np.load(f, allow_pickle=True)
        ep = {k: v for k, v in ep.items()}
        frames = ep["low"]  # cant use side frame rn

        # frame = cu.square(frame)
        # frame = cv2.resize(frame, (224, 224))

        n = len(frames)
        steps = []
        for frame in tqdm(frames, leave=False):
            pack = {"img": frame}
            out = client.infer(pack)
            if out is None:
                continue
            out = postprocess(out, frame)
            img = out.pop("img")
            wrist = out.pop("img_wrist")

            step = {
                "observation": {
                    "image": {"low": img, "wrist": wrist},
                    "state": out,
                }
            }
            steps.append(step)

        try:
            steps = jax.tree.map(lambda *_x: np.stack(_x, axis=0), *steps)
        except Exception as e:
            logger.error(f"Error stacking steps: {e}")

            shapes = jax.tree.map(lambda *_x: set(__x.shape for __x in _x), *steps)
            pprint(shapes)
            raise e

        pprint(spec(steps))
        # quit()

        obs = steps.pop("observation")

        img = obs.pop("image")
        img = jax.tree.map(lambda x: x / 255, img)  # to encode as mp4 with ffmpeg
        obs["image"] = img

        steps["observation"] = obs

        lang = {
            "lang": np.array([_lang for _ in range(n)]),
            "task": _task,
        }
        steps["lang"] = lang["lang"]

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

        if dataset is None:
            pprint(spec(step0))
            dataset = create(
                repo_id=cfg.repo_id,
                robot_type="human",
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
        # **({'branch':cfg.branch} if cfg.branch else {}),
        tags=tags,
        private=False,
        push_videos=True,
    )
    # for i, s in tqdm(enumerate(steps), total=len(steps), leave=False):


if __name__ == "__main__":
    main(tyro.cli(MyConfig))
