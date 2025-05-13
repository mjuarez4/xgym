import enum
from dataclasses import field
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
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rich.pretty import pprint
from tqdm import tqdm

import xgym
from xgym.lrbt.convert import (
    DEFAULT_DATASET_CONFIG,
    FRAME_MODE,
    MOTORS,
    DatasetConfig,
    Embodiment,
    Task,
    create,
)
from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose
from xgym.rlds.util.trajectory import binarize_gripper_actions, scan_noop
from xgym.viz.mano import overlay_palm, overlay_pose

from xgym.lrbt.from_memmap import Config


import time
from enum import Enum

from tqdm import tqdm

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import jax
import numpy as np
import tyro

# from get_calibration import MyCamera
# from mano_pipe_v3 import remap_keys, select_keys
from rich.pretty import pprint
from webpolicy.deploy.client import WebsocketClientPolicy
from typing import Literal

from xgym import BASE
from xgym.calibrate.april import Calibrator
from xgym.rlds.util import (
    add_col,
    apply_persp,
    apply_uv,
    apply_xyz,
    perspective_projection,
    remove_col,
    solve_uv2xyz,
)

np.set_printoptions(suppress=True, precision=3)

def spec(thing: dict[str, np.ndarray]):
    """Returns the shape of each key in the dict."""
    return jax.tree.map(lambda x: x.shape, thing)


import threading
from typing import Union


class MyCamera:
    def __repr__(self) -> str:
        return f"MyCamera(device_id={'TODO'})"

    def __init__(self, cam: Union[int, cv2.VideoCapture]):
        self.cam = cv2.VideoCapture(cam) if isinstance(cam, int) else cam
        self.thread = None

        self.fps = 30
        self.dt = 1 / self.fps

        # while
        # _, self.img = self.cam.read()
        self.start()
        time.sleep(1)

    def start(self):

        self._recording = True

        def _record():
            while self._recording:
                tick = time.time()
                # ret, self.img = self.cam.read()
                self.cam.grab()

                toc = time.time()
                elapsed = toc - tick
                time.sleep(max(0, self.dt - elapsed))

        self.thread = threading.Thread(target=_record, daemon=True)
        self.thread.start()

    def stop(self):
        self._recording = False
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            del self.thread

    def read(self):
        ret, img = self.cam.retrieve()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ret, img


class NPZVideoReader:

    def __init__(self, dir: str):
        self.dir = dir
        self.files = list(Path(dir).rglob("ep*.npz"))
        self._load_next_episode()

    def _load_next_episode(self):
        if not self.files:
            self.frames = None
            return
        f = self.files.pop(0)
        ep = np.load(f, allow_pickle=True)
        ep = {k: ep[k] for k in ep.keys()}
        pprint(spec(ep))  # assuming spec() is defined elsewhere
        self.frames = ep[list(ep.keys())[0]]
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.frames is None:
            raise StopIteration
        if self.idx >= self.frames.shape[0]:
            self._load_next_episode()
            return self.__next__()
        frame = self.frames[self.idx]
        self.idx += 1
        return True, frame

    def read(self):
        return self.__next__()

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
class Source:
    pass

@dataclass
class Camera(Source):
    idx: int=3 # the camera idx to use

@dataclass
class File(Source):
    dir: str 

    # typ: Literal["mp4", "npz"] = "npz" # the type of video file

@dataclass
class HamerConfig:
    host: str
    port: int

    # src: Camera | File 
    # cam: int = 3  # the camera idx to use
    # mode: Mode = Mode.CAM  # the mode to use, either CAM or FILE


def remap_keys(out: dict):
    out = {k.replace("pred_", "").replace("_params", ""): v for k, v in out.items()}
    return out


def select_keys(out: dict):
    """Selects keys to keep in the output.
    as a side effect, prepares the keypoints_3d for further processing
    TODO break into separate func?
    """

    out["keypoints_3d"] += out.pop("cam_t_full")[:, None, :]
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
        "keypoints_2d",  # these are wrt the box not full img ... solve for 2d
        "keypoints_3d",
        "mano.betas",
        "mano.global_orient",
        "mano.hand_pose",
        # "vertices",
        "right",
        # "focal_length", # i think we use the scaled one instead
        "scaled_focal_length",
    ]
    return {k: out[k] for k in keep if k in out}

def postprocess(out:dict,frame:np.ndarray):

    out = jax.tree.map(lambda x: x.copy(), out)
    out = remap_keys(out)
    box = {
        "center": out["box_center"][0],
        "size": out["box_size"][0],
        "kp2d": out.pop("keypoints_2d")[0], # only relevant to box
    }

    right = bool(out["right"][0])
    left = not right

    if left:
        n = len(out["keypoints_3d"])
        kp3d = add_col(out["keypoints_3d"])
        kp3d = remove_col((kp3d[0] @ T_xflip)[None])
        out["keypoints_3d"] = np.concatenate([kp3d for _ in range(n)])

    out = select_keys(out)

    f = out["scaled_focal_length"]
    P = perspective_projection(f, H=frame.shape[0], W=frame.shape[1])
    points2d = apply_persp(out["keypoints_3d"], P)[0, :, :-1]
    out['kp2d'] = points2d
    out['kp3d'] = out.pop("keypoints_3d")

    out = out | {'box': box}
    squeeze = lambda arr: jax.tree.map(lambda x: x.squeeze(), arr)
    out = squeeze(out)

    def maybe_unsqueeze(x):
        return x.reshape((-1)) if x.ndim <= 1 else x
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

    taskfile = next(Path(cfg.dir).glob("*.npy"))
    _task = taskfile.stem.replace("_", " ")
    _lang = np.load(taskfile)

    pprint(cfg)

    dataset = None
    for f in tqdm(ds, total=len(ds)):

        client = WebsocketClientPolicy(host=cfg.hamer.host, port=cfg.hamer.port)
        ep = np.load(f, allow_pickle=True)
        ep = {k: v for k, v in ep.items()}
        frames = ep['low'] # cant use side frame rn

        # frame = cu.square(frame)
        # frame = cv2.resize(frame, (224, 224))

        n = len(frames)
        steps = []
        for frame in tqdm(frames,leave=False):
            pack = {"img": frame}
            out = client.infer(pack)
            if out is None:
                continue
            out = postprocess(out, frame)
            img = out.pop("img")
            wrist = out.pop("img_wrist")

            step = {
                "observation": {
                    "image": { 'low': img, 'wrist': wrist },
                    "state":  out ,
                }
            }
            steps.append(step)


        steps = jax.tree.map(lambda *_x: np.stack(_x, axis=0), *steps)

        pprint(spec(steps))
        # quit()

        obs = steps.pop("observation")

        img = obs.pop("image")
        img = jax.tree.map(lambda x: x / 255, img) # to encode as mp4 with ffmpeg
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
