import glob
import json
import os
import os.path as osp
from typing import Any, Iterator, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


def downscale_to_224(height: int, width: int) -> Tuple[int, int]:
    """
    Downscale the image so that the shorter dimension is 224 pixels,
    and the longer dimension is scaled by the same ratio.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        Tuple[int, int]: The new height and width of the image.
    """
    # Determine the scaling ratio
    if height < width:
        ratio = 224.0 / height
        new_height = 224
        new_width = int(width * ratio)
    else:
        ratio = 224.0 / width
        new_width = 224
        new_height = int(height * ratio)

    return new_height, new_width


class XgymLiftSingle(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for LUC XGym Single Arm v1.0.1"""

    VERSION = tfds.core.Version("1.0.1")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.0.1": "Non blocking at 5hz... 3 world cams",
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
                                    "image": tfds.features.FeaturesDict(
                                        {
                                            "camera_0": tfds.features.Image(
                                                shape=(224, 224, 3),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Main camera RGB observation.",
                                            ),
                                            "camera_1": tfds.features.Image(
                                                shape=(224, 224, 3),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Main camera RGB observation.",
                                            ),
                                            "camera_2": tfds.features.Image(
                                                shape=(224, 224, 3),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Main camera RGB observation.",
                                            ),
                                            "wrist": tfds.features.Image(
                                                shape=(224, 224, 3),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Wrist camera RGB observation.",
                                            ),
                                        }
                                    ),
                                    "proprio": tfds.features.FeaturesDict(
                                        {
                                            "joints": tfds.features.Tensor(
                                                shape=[7],
                                                dtype=np.float32,
                                                doc="Joint angles. radians",
                                            ),
                                            "position": tfds.features.Tensor(
                                                shape=[7],
                                                dtype=np.float32,
                                                doc="Joint positions. xyz millimeters (mm) and rpy",
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                # do we need 8? for terminate episode action?
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [xyz,rpy,gripper].",
                            ),
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

        root = osp.expanduser("~/tensorflow_datasets/xgym_single/source")
        root = osp.expanduser("~/ducks/xgym-sandbox*")

        files = glob.glob(root)
        files = [f for f in files if any([x.endswith("npz") for x in os.listdir(f)])]

        self.filtered = osp.expanduser("~/data/filtered.json")

        return {
            "train": self._generate_examples(files),
        }

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

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )
        task = "pick up ducks"  # hardcoded for now
        lang = self._embed([task])[0].numpy()  # embedding takes ≈0.06s

        def _parse_example(idx, ep):

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            ep = np.load(ep)
            ep = self.dict_unflatten({x: ep[x] for x in ep.files})

            if ep["robot"]["joints"].shape[1] != 7:
                ep["robot"]["joints"] = ep["robot"]["joints"].T  # patch

            ep["proprio"] = jax.tree.map(
                lambda x: x.astype(np.float32), ep.pop("robot")
            )

            ep["image"] = jax.tree_map(
                lambda x: tf.image.resize(x, (224, 224)).numpy().astype(np.uint8),
                ep.pop("img"),
            )

            # patch
            ep["image"] = {
                **{"wrist": ep["image"]["wrist"]},
                **{
                    f"camera_{i}": ep["image"][x]
                    for i, x in enumerate(
                        [y for y in ep["image"].keys() if "camera" in y]
                    )
                },
            }

            episode = []
            n = len(ep["proprio"]["position"])

            spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)
            # print(spec(ep))

            prevact = None
            for i in range(n - 1):

                prev = jax.tree.map(lambda x: x[i - 1], ep) if i > 0 else None
                step = jax.tree.map(lambda x: x[i], ep)
                next = jax.tree.map(lambda x: x[i + 1], ep)

                act = next["proprio"]["position"] - step["proprio"]["position"]
                act[:3] = act[:3] / int(1e3)
                #act[3:6] = 0
                act[-1] = step["proprio"]["position"][-1] / 850

                # if norm is less than eps
                if self.is_noop(act, prevact):
                    continue

                prevact = act

                episode.append(
                    {
                        "observation": step,
                        "action": act.astype(np.float32),
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

        for path in paths:
            ds = [osp.join(path, x) for x in os.listdir(path) if x.endswith("npz")]

            for idx, ep in enumerate(ds):
                yield _parse_example(f"{path}_{idx}", ep)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(ds) | beam.Map(_parse_example)
