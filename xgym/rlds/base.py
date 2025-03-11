import abc
from abc import ABC
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from rich.pretty import pprint

import xgym
from xgym.rlds.util.trajectory import binarize_gripper_actions as binarize
from xgym.rlds.util.trajectory import scan_noop



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class TFDSBaseMano(tfds.core.GeneratorBasedBuilder, ABC):
    """DatasetBuilder Base Class for LUC XGym Mano"""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "3.0.0": "New location.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "img": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "img_wrist": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="RGB observation that follows the hand.",
                                    ),
                                    # "focal_length": tfds.features.Tensor( shape=(), dtype=np.float32),
                                    "scaled_focal_length": tfds.features.Tensor(
                                        shape=(), dtype=np.float32
                                    ),
                                    "keypoints_2d": tfds.features.Tensor(
                                        shape=(21, 2),
                                        dtype=np.float32,
                                        doc="2D keypoints in pixel space.",
                                    ),
                                    "keypoints_3d": tfds.features.Tensor(
                                        shape=(21, 3),
                                        dtype=np.float32,
                                        doc="3D keypoints in camera space.",
                                    ),
                                    "mano.betas": tfds.features.Tensor(
                                        shape=(10,),
                                        dtype=np.float32,
                                        doc="Hand shape parameters.",
                                    ),
                                    "mano.global_orient": tfds.features.Tensor(
                                        shape=(1, 3, 3),
                                        dtype=np.float32,
                                        doc="Global orientation of the hand.",
                                    ),
                                    "mano.hand_pose": tfds.features.Tensor(
                                        shape=(15, 3, 3),
                                        dtype=np.float32,
                                        doc="Hand pose parameters.",
                                    ),
                                    "right": tfds.features.Tensor(
                                        shape=(),
                                        dtype=np.float32,
                                        doc="True if right hand.",
                                    ),
                                }
                            ),
                            # "action": tfds.features.Tensor(
                            # shape=(7,),
                            # dtype=np.float32,
                            # doc="Robot action, consists of [xyz,rpy,gripper].",
                            # ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict({}),
                }
            )
        )

    def _split_helper(self, loc: str):
        """helper function to look in DATA for npz files"""
        from xgym import DATA

        files = (Path(DATA) / loc).expanduser().glob("*.npz")  # episodes are npz
        files = list(files)
        xgym.logger.info(f"Found {len(files)} episodes.")
        return files

    @abc.abstractmethod
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""

        raise NotImplementedError("Implement this method in your subclass.")
        files = self._split_helper("xgym_duck_mano")
        return {"train": self._generate_examples(files)}

    def is_noop(self, action, prev_action=None, threshold=1e-3):
        """
        Returns whether an action is a no-op action.

        A no-op action satisfies two criteria:
            (1) All action dimensions, except for the last one (gripper action), are near zero.
            (2) The gripper action is equal to the previous timestep's gripper action.

        Explanation of (2):
            Naively filtering out actions with just criterion (1) is not good because you will
            remove actions where the robot is staying still but opening/closing its gripper.
            So you also need to consider the current state (by checking the previous timestep's
            gripper action as a proxy) to determine whether the action really is a no-op.
        """

        # Special case: Previous action is None if this is the first action in the episode
        # Then we only care about criterion (1)
        if prev_action is None:
            return np.linalg.norm(action[:-1]) < threshold

        # Normal case: Check both criteria (1) and (2)
        gripper_action = action[-1]
        prev_gripper_action = prev_action[-1]
        return (
            np.linalg.norm(action[:-1]) < threshold
            and gripper_action == prev_gripper_action
        )

    def dict_unflatten(self, flat, sep="."):
        """Unflatten a flat dictionary to a nested dictionary."""

        nest = {}
        for key, value in flat.items():
            keys = key.split(sep)
            d = nest
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
        return nest

    def to_rgb(self, ep: Dict):
        """Converts Union[RGB,BGR] to RGB images."""

        video = ep["img_wrist"]
        channels = np.median(video, axis=(0, 1, 2))
        # wchannels = ep["img_wrist"].mean(-2).mean(-2).mean(0)
        flipc = channels[0] < channels[-1]
        # print(channels)
        # print(flipc)
        if flipc:
            # xgym.logger.info("Flipping BGR to RGB")
            ep["img"] = ep["img"][..., ::-1]
            ep["img_wrist"] = ep["img_wrist"][..., ::-1]

        return ep

    def _generate_examples(self, ds) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # task = "embodiment:Human, task:pick up the red block"  # hardcoded for now
        taskfile = next(Path().cwd().glob("*.npy"))
        task = taskfile.stem.replace("_", " ")
        lang = np.load(taskfile)

        def _parse_example(idx, ep):

            ep = np.load(ep)
            ep = {k: ep[k] for k in ep.files}
            # ep = self.dict_unflatten({k: ep[k] for k in ep.files})

            # pprint(spec(ep))
            # quit()

            ep = {
                k: v.astype(np.float32) if "img" not in k else v for k, v in ep.items()
            }
            ep = self.to_rgb(ep)

            next = None
            episode = []  # last step is used for noop
            for i, step in enumerate(ep["img"][:-1]):

                # prev = jax.tree.map(lambda x: x[i - 1], ep) if i > 0 else None
                step = jax.tree.map(lambda x: x[i], ep) if next is None else next
                next = jax.tree.map(lambda x: x[i + 1], ep)

                # act is only for finding noop
                act = next["keypoints_2d"] - step["keypoints_2d"]

                # not moving more than <threshold>px
                if self.is_noop(act, None, threshold=3):
                    continue

                episode.append(
                    {
                        "observation": step,
                        # "action": act.astype(np.float32), # actions are on the fly for mano
                        "discount": 1.0,
                        "reward": float(i == (len(ep) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(ep) - 1),
                        "is_terminal": i == (len(ep) - 1),
                        "language_instruction": task,
                        "language_embedding": lang,
                    }
                )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {}}

            # if you want to skip an example for whatever reason, simply return None
            return idx, sample

        for idx, ep in enumerate(ds):
            yield _parse_example(f"{ep}_{idx}", ep)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(ds) | beam.Map(_parse_example)



class XgymSingle(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder Base for LUC XGym"""

    # VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "2.0.0": "more data and overhead cam",
        "3.0.0": "relocated setup",
        "4.0.0": "50hz data",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""

        def feat_im(doc):
            return tfds.features.Image(
                shape=(224, 224, 3),
                dtype=np.uint8,
                encoding_format="png",
                doc=doc,
            )

        def feat_prop():
            return tfds.features.FeaturesDict(
                {
                    "joints": tfds.features.Tensor(
                        shape=[7],
                        dtype=np.float32,
                        doc="Joint angles. radians",
                    ),
                    "position": tfds.features.Tensor(
                        shape=[6],
                        dtype=np.float32,
                        doc="Joint positions. xyz millimeters (mm) and rpy",
                    ),
                    "gripper": tfds.features.Tensor(
                        shape=[1],
                        dtype=np.float32,
                        doc="Gripper position. 0-850",
                    ),
                }
            )

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.FeaturesDict(
                                        {
                                            "worm": feat_im(
                                                doc="Low front logitech camera RGB observation."
                                            ),
                                            "side": feat_im(
                                                doc="Low side view logitech camera RGB observation."
                                            ),
                                            "overhead": feat_im(
                                                doc="Overhead logitech camera RGB observation."
                                            ),
                                            "wrist": feat_im(
                                                doc="Wrist realsense camera RGB observation."
                                            ),
                                        }
                                    ),
                                    "proprio": feat_prop(),
                                }
                            ),
                            "action": feat_prop(),  # TODO does it make sense to store proprio and  actions?
                            #
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict({}),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""

        root = Path.home() / f"{self.name}_v{ str(self.VERSION)[0]}"
        files = list(root.rglob("*.dat"))
        return {"train": self._generate_examples(files)}

    def dict_unflatten(self, flat, sep="."):
        """Unflatten a flat dictionary to a nested dictionary."""

        nest = {}
        for key, value in flat.items():
            keys = key.split(sep)
            d = nest
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
        return nest

    def _parse_example(self, path):

        try:
            info, ep = xgym.viz.memmap.read(path)
        except Exception as e:
            xgym.logger.error(f"Error reading {path}")
            xgym.logger.error(e)
            return None

        n = len(ep["time"])

        # pprint(self.spec(ep))

        ### cleanup and remap keys
        ep.pop("time")
        ep.pop("gello_joints")

        ep["robot"] = {}
        ep["robot"]["joints"] = ep.pop("xarm_joints")
        ep["robot"]["position"] = ep.pop("xarm_pose")
        ep["robot"]["gripper"] = ep.pop("xarm_gripper")
        # ep['robot'] = {'joints': joints, 'position': np.concatenate((pose, grip), axis=1)}

        try:  # we dont want the ones with only rs
            _ = ep.get("/xgym/camera/worm")
        except KeyError:
            return None

        zeros = lambda: np.zeros((n, 224, 224, 3), dtype=np.uint8)
        ep["/xgym/camera/wrist"] = ep.pop("/xgym/camera/rs")
        ep["/xgym/camera/overhead"] = ep.pop("/xgym/camera/over", zeros())
        ep["image"] = {
            k: ep.pop(f"/xgym/camera/{k}", zeros())
            for k in ["worm", "side", "overhead", "wrist"]
        }

        ### scale and binarize
        ep["robot"]["gripper"] /= 850
        ep["robot"]["position"][:, :3] /= 1e3
        pbin = partial(binarize, open=0.95, close=0.4)  # doesnt fully close
        ep["robot"]["gripper"] = np.array(pbin(jnp.array(ep["robot"]["gripper"])))
        pos = np.concatenate((ep["robot"]["position"], ep["robot"]["gripper"]), axis=1)

        ### filter noop
        noops = np.array(scan_noop(jnp.array(pos), threshold=1e-3))
        mask = ~noops
        ep = jax.tree.map(select := lambda x: x[mask], ep)

        ### calculate action
        action = jax.tree.map(
            lambda x: x[1:] - x[:-1], ep["robot"]
        )  # pose and joint action
        action["gripper"] = ep["robot"]["gripper"][1:]  # gripper is absolute
        ep = jax.tree.map(lambda x: x[:-1], ep)
        # ep["action"] = action # action is not an observation

        # pprint(self.spec(ep))

        ### final remaps
        ep["proprio"] = ep.pop("robot")

        geti = lambda x, i: jax.tree.map(lambda y: y[i], x)
        episode = [
            {
                "observation": geti(ep, i),
                "action": geti(action, i),
                "discount": 1.0,
                "reward": float(i == (len(ep) - 1)),
                "is_first": i == 0,
                "is_last": i == (len(ep) - 1),
                "is_terminal": i == (len(ep) - 1),
                "language_instruction": self.task,
                "language_embedding": self.lang,
            }
            for i in range(len(ep["proprio"]["position"]))
        ]

        # if you want to skip an example for whatever reason, simply return None
        sample = {"steps": episode, "episode_metadata": {}}
        id = f"{path.parent.name}_{path.stem}"
        return id, sample

    def _generate_examples(self, ds) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        self.taskfile = next(Path().cwd().glob("*.npy"))
        self.task = self.taskfile.stem.replace("_", " ")
        self.lang = np.load(self.taskfile)

        for path in ds[1:]:
            ret =  self._parse_example(path)
            if ret is not None:
                yield ret

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(ds) | beam.Map(_parse_example)
