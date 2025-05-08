import enum
from copy import deepcopy

import torch
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import cv2
import draccus
import jax
import jax.numpy as jnp
import lerobot
import numpy as np
import tensorflow_datasets as tfds
import tyro
from flax.traverse_util import flatten_dict
from jax import numpy as jnp
from lerobot.common.datasets.lerobot_dataset import \
    HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rich.pretty import pprint
from tqdm import tqdm

import xgym
from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose
from xgym.rlds.util.trajectory import binarize_gripper_actions, scan_noop
from xgym.viz.mano import overlay_palm, overlay_pose


class Task(enum.Enum):
    LIFT = "lift"
    STACK = "stack"
    DUCK = "duck"
    POUR = "pour"


class Embodiment(enum.Enum):
    SINGLE = "single"
    MANO = "mano"


@dataclass
class BaseReaderConfig:

    embodiment: Embodiment = Embodiment.SINGLE
    task: Task = Task.LIFT
    version: str = tyro.MISSING
    verbose: bool = False
    visualize: bool = True

    use_palm: bool = False
    wait: int = 20  # the cv2 wait time in ms
    horizon: int = 10
    resolution: int = 3

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, enum.Enum):
                setattr(self, k, v.value)


@dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False # True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 16
    image_writer_threads: int = 8
    video_backend: str | None = None # 'h264_nvenc'


DEFAULT_DATASET_CONFIG = DatasetConfig()
FRAME_MODE = 'video' if DEFAULT_DATASET_CONFIG.use_videos else 'image'

MOTORS = {
    "aloha": [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ],
    "xarm": [f"joint_{i}" for i in range(1, 8)] + ["gripper"],
}


def create(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = FRAME_MODE,
    *,
    example: dict[str, np.ndarray],
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """defines and creates the dataset using the first step"""

    motors = MOTORS[robot_type]

    cameras = example["observation"]["image"].keys()

    class Name(enum.Enum):
        WHC = ["width", "height", "channels"]
        CWH = ["channels", "width", "height"]

        GRIPPER = ["gripper"]
        DOF7 = [f"joint_{i}" for i in range(1, 8)]
        DOF6 = [f"joint_{i}" for i in range(1, 7)]

        POSE = ["x", "y", "z", "rx", "ry", "rz"]
        POSEQ = ["x", "y", "z", "qw", "qx", "qy", "qz"]

        LANG = None

    NAMES = {
        "observation": {
            "image": {
                "high": Name.WHC,
                "low": Name.WHC,
                "side": Name.WHC,
                "wrist": Name.WHC,
            },
            "state": {
                "gripper": Name.GRIPPER,
                "joints": Name.DOF7,
                "position": Name.POSE, # TODO fix
                # "pose": Name.POSE, # throw hf error?
            },
        },
        "lang": Name.LANG,
    }

    def make_spec(k, arr):
        key = ".".join([str(_k.key) for _k in k])
        dtype = arr.dtype.name if "image" not in key else mode
        return {
            "dtype": dtype,
            "shape": arr.shape,
            "names": flatten_dict(NAMES, sep=".")[key].value,
        }

    features = flatten_dict(example, sep=".")
    features = jax.tree.map_with_path(make_spec, features)
    # pprint(features)

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def convert(
    dataset: LeRobotDataset,
    files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:

    files = Path("~/repo/robosuite/dataset").expanduser().glob("*.hdf5")
    files = list(files)

    for file in tqdm(files[:5], total=len(files)):
        # ep = np.load(file, allow_pickle=True)
        # ep = {k: ep[k] for k in ep.files}

        def extract_all_data_as_dict(file_path):
            data = {}
            with h5py.File(file_path, "r") as f:

                def recursive_extract(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name] = obj[()]  # Load as np.ndarray

                f.visititems(recursive_extract)
            return data

        spec = lambda arr: jax.tree.map(lambda _x: _x.shape, arr)
        steps = extract_all_data_as_dict(file)

        # steps = jax.tree.map(lambda *_x: np.stack([_x], axis=0), *ep["steps"])
        # pprint(spec(steps))

        nsteps = steps["action"].shape[0]

        grip = steps["action"][:, -1]
        grip = np.concatenate([[grip[0]], grip[:-1]])
        qpos = np.concatenate([steps["observations/qpos"], grip[:, None]], axis=-1)

        out = {
            "observation": {
                "state": qpos,
                # "velocity": steps["observations/qvel"],
                # "effort":
                "images": {
                    "high": steps["observations/images/high"],
                    "low": steps["observations/images/low"],
                    "wrist": steps["observations/images/wrist"],
                },
            },
            "action": steps["action"],
            # "task": task,
        }
        out = flatten_dict(out, sep=".")

        for k in [
            "observation.images.high",
            "observation.images.low",
            "observation.images.wrist",
        ]:
            out[k] = (out[k] / 255).astype(np.float32)

        for n in range(nsteps):
            frame = jax.tree.map(lambda x: torch.from_numpy(x[n]).float(), out)
            # pprint(spec(frame))
            dataset.add_frame(frame | {"task": task})

        dataset.save_episode()

    return dataset


def main(cfg: BaseReaderConfig):
    name = f"xgym_{cfg.task}_{cfg.embodiment}:{cfg.version}"
    xgym.logger.info(f"Loading {name} dataset")
    ds = tfds.load(name)["train"]

    spec = lambda tree: jax.tree.map(lambda x: x.shape, tree)
    take = lambda tree, _i: jax.tree.map(lambda x: x[_i], tree)

    dataset = None
    for ep in tqdm(ds, total=len(ds)):
        n = len(ep["steps"])
        steps = [x for x in ep["steps"]]
        steps = jax.tree.map(lambda x: np.array(x), steps)
        steps = jax.tree.map(lambda *_x: np.stack(_x, axis=0), *steps)

        steps.pop("discount")
        steps.pop("reward")
        steps.pop("is_first")
        steps.pop("is_last")
        steps.pop("is_terminal")
        steps.pop("action")

        obs = steps.pop("observation")
        img = obs.pop("image")
        img["high"] = img.pop("overhead")
        img["low"] = img.pop("worm")
        # to encode as mp4 with ffmpeg
        img = jax.tree.map(lambda x: x / 255, img)
        obs["image"] = img

        # force to be joint actions
        state = obs.pop("proprio")
        state = np.concatenate([state["joints"],state['gripper']], axis=-1)
        obs["state"] = state
        steps['action'] = state
        pprint(steps["action"].shape)

        steps["observation"] = obs

        lang = {
            "lang": steps.pop("language_embedding"),
            "task": steps.pop("language_instruction"),
        }
        steps["lang"] = lang["lang"]



        # pprint(spec(steps))
        step0 = take(steps, 0)



        if dataset is None:
            pprint(spec(step0))
            dataset = create(
                repo_id="mhyatt000/demo",
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
        branch="v0.1",
        tags=tags,
        private=False,
        push_videos=True,
    )
    # for i, s in tqdm(enumerate(steps), total=len(steps), leave=False):


if __name__ == "__main__":
    main(tyro.cli(BaseReaderConfig))
