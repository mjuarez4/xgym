import shutil
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, TypeVar, overload

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm
from xgym import MANO_1, MANO_1DONE, MANO_2
from xgym.controllers import HamerController


def stack(seq: list[dict]):
    """Stacks all frames in the seq into arrays for saving."""

    stacked = {}
    for k in seq[0].keys():
        print(k)
        stacked[k] = np.stack([s[k] for s in seq], axis=0)
    return stacked

    stacked = {k: np.stack([s[k] for s in seq], axis=0) for k in seq[0].keys()}
    return stacked


def spec(thing: dict[str, np.ndarray]):
    return jax.tree.map(lambda x: x.shape, thing)


example = {
    "box_center": (1, 2),
    "box_size": (1,),
    "focal_length": (1, 2),
    "img": (1, 3, 256, 256),
    "img_size": (1, 2),
    "personid": (1,),
    "pred_cam": (1, 3),
    "pred_cam_t": (1, 3),
    "pred_cam_t_full": (1, 3),
    "pred_keypoints_2d": (1, 21, 2),
    "pred_keypoints_3d": (1, 21, 3),
    "pred_mano_params.betas": (1, 10),
    "pred_mano_params.global_orient": (1, 1, 3, 3),
    "pred_mano_params.hand_pose": (1, 15, 3, 3),
    "pred_vertices": (1, 778, 3),
    "right": (1,),
    "scaled_focal_length": (),
}


def postprocess_sequence(seq: list[dict]):
    """Filters frames to ensure only right-hand detections are included.
    returns:
        dict: the filtered seq
        bool: whether the seq is usable
    """

    is_right = np.array([s["right"].mean() for s in seq]).mean()
    is_right = is_right > 0.5

    if not is_right:  # skip if predominantly left hand
        return {}, False

    def select_hand(s):

        # if x looks like the example then not batched
        not_batched = all([s[k].shape == example[k] for k in s.keys()])
        not_batched = True # squeeze helped

        def _select(x):
            if not_batched:
                return x[is_right]
            else:
                return x[0, is_right]
            return x

        return jax.tree.map(_select, s)

    out = [select_hand(s) for s in seq]
    return out, True


def main():

    hamer = HamerController("aisec-102.cs.luc.edu", 8001)

    files = list(MANO_1.glob("*.npz"))
    print(files)

    files = [f for f in files if "7" in f.name]

    for path in tqdm(files, leave=False):
        print(f"Processing video: {path.name}")

        data = np.load(path)
        data = {k: data[k] for k in data.files}

        for k, v in tqdm(data.items(), leave=False):  # for each view in the episode
            outs = []
            for i, frame in tqdm(enumerate(v[45:50]), total=len(v)):

                cv2.imshow("frame", frame)
                cv2.waitKey(1)

                out = hamer(frame)
                {k:v.squeeze() for k, v in out.items()} 
                pprint(spec(out))
                if out is None or out == {}:
                    continue
                print(out["right"].mean())
                outs.append(out)

            outs, ok = postprocess_sequence(outs)
            if not ok:
                continue

            # list of dict to dict of stacked np arrays
            pprint(spec(outs))
            outs = stack(outs)
            np.savez(MANO_2 / f"{path.stem}_{k}_filtered.npz", **outs)

        # shutil.move(str(path), MANO_1DONE / path.name)


if __name__ == "__main__":
    main()
