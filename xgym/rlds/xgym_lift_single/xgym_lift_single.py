from pathlib import Path
from typing import Any, Iterator, Tuple

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class XgymLiftSingle(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for LUC XGym Single Arm"""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.0.1": "Non blocking at 5hz... 3 world cams",
        "2.0.0": "teleoperated demos... high cam",
        "3.0.0": "relocated out of sun ... renamed/repositioned cams",
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
                                            "worm": tfds.features.Image(
                                                shape=(224, 224, 3),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Low front logitech camera RGB observation.",
                                            ),
                                            "side": tfds.features.Image(
                                                shape=(224, 224, 3),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Low side view logitech camera RGB observation.",
                                            ),
                                            "overhead": tfds.features.Image(
                                                shape=(224, 224, 3),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Overhead logitech camera RGB observation.",
                                            ),
                                            "wrist": tfds.features.Image(
                                                shape=(224, 224, 3),
                                                dtype=np.uint8,
                                                encoding_format="png",
                                                doc="Wrist realsense camera RGB observation.",
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

        files = list(Path("~/xgym_lift3").expanduser().rglob("*.npz"))
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

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        taskfile = next(Path().cwd().glob("*.npy"))
        task = taskfile.stem.replace("_", " ")
        lang = np.load(taskfile)

        def _parse_example(ep):

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            ep = np.load(str(path))
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

            episode = []
            n = len(ep["proprio"]["position"])

            spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)
            # print(spec(ep))
            # quit()

            prevact = None
            for i in range(n - 1):

                prev = jax.tree.map(lambda x: x[i - 1], ep) if i > 0 else None
                step = jax.tree.map(lambda x: x[i], ep)
                next = jax.tree.map(lambda x: x[i + 1], ep)

                act = next["proprio"]["position"] - step["proprio"]["position"]
                act[:3] = act[:3] / int(1e3)
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

            # if you want to skip an example for whatever reason, simply return None
            sample = {"steps": episode, "episode_metadata": {}}
            id = f"{path.parent.name}_{path.stem}"
            return id, sample

        for path in paths:
            yield _parse_example(path)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(ds) | beam.Map(_parse_example)
