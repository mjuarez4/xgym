from pathlib import Path
from pprint import pprint
from typing import Any, Iterator, Tuple

import jax
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm


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


class XgymLiftMano(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for LUC XGym Mano v1.0.0"""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
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
                                    "frame": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    # "frame_human0": tfds.features.Image(
                                    # shape=(224, 512, 3),
                                    # dtype=np.uint8,
                                    # encoding_format="png",
                                    # doc="RGB observation that follows the hand.",
                                    # ),
                                    "focal_length": tfds.features.Tensor(
                                        shape=(), dtype=np.float32
                                    ),
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
                                    "mano": tfds.features.FeaturesDict(
                                        {
                                            "betas": tfds.features.Tensor(
                                                shape=(10,),
                                                dtype=np.float32,
                                                doc="Hand shape parameters.",
                                            ),
                                            "global_orient": tfds.features.Tensor(
                                                shape=(1,3,3,),
                                                dtype=np.float32,
                                                doc="Global orientation of the hand.",
                                            ),
                                            "hand_pose": tfds.features.Tensor(
                                                shape=(15,3,3,),
                                                dtype=np.float32,
                                                doc="Hand pose parameters.",
                                            ),
                                        }
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

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""

        root = Path("~/xgym_mano").expanduser().glob("*_tfds")  # episodes are npz
        root = Path("~/.bala/output_hamer").expanduser().glob("*_tfds*")  # episodes are npz

        return { "train": self._generate_examples(root) }

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

    def _generate_examples(self, ds) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # self._embed = hub.load( "https://tfhub.dev/google/universal-sentence-encoder-large/5")
        task = "embodiment:Human, task:pick up the red block"  # hardcoded for now
        # lang = self._embed([task])[0].numpy()  # embedding takes ≈0.06s
        lang = np.zeros(512, dtype=np.float32)

        self.obskeys = [
            "focal_length",
            "scaled_focal_length",
            #
            "frame",
            "keypoints_2d",
            "keypoints_3d",
            "mano",
            # "mano.betas",
            # "mano.global_orient",
            # "mano.hand_pose",
            "right",
        ]

        def _parse_example(idx, ep):

            # sort series of npz files (steps in episode) by index
            # ep = ep.glob("*.npz")
            # ep = list(
                # sorted(ep, key=lambda x: int(x.name.split(".")[0].split("_")[-1]))
            # )

            spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)

            ep = np.load(ep)
            ep = self.dict_unflatten({k: ep[k] for k in ep.files})
            ep['mano'] = ep.pop('mano_params')

            pprint(spec(ep))
            ep = {k:v for k,v in ep.items() if k in self.obskeys}

            frame = ep.pop("frame")
            ep = jax.tree.map(lambda x: x.astype(np.float32), ep)
            ep["frame"] = frame

            ep['keypoints_2d'] = ep['keypoints_2d'][:, :, :-1]  # remove visibility
            pprint(spec(ep))


            next = None
            episode = []  # last step is used for noop
            for i, step in tqdm(enumerate(ep['frame'][:-1]), total=len(ep)):

                # prev = jax.tree.map(lambda x: x[i - 1], ep) if i > 0 else None
                step = jax.tree.map(lambda x: x[i] , ep) if next is None else next
                next = jax.tree.map(lambda x: x[i + 1], ep) 

                # act is only for finding noop
                act = next["keypoints_2d"] - step["keypoints_2d"]

                # not moving more than 1px
                if self.is_noop(act, None, threshold=3):
                    # print("noop")
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
