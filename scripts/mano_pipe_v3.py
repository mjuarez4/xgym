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
from xgym import MANO_1, MANO_1DONE, MANO_2, MANO_4
from xgym.model_controllers import HamerController
from xgym.rlds.util import (add_col, apply_persp, apply_uv, apply_xyz,
                            perspective_projection, remove_col, solve_uv2xyz)
from xgym.rlds.util.render import render_openpose
from xgym.rlds.util.transform import center_crop


def stack(seq: list[dict]):
    """Stacks all frames in the seq into arrays for saving."""
    stacked = {}
    for k in seq[0].keys():
        stacked[k] = np.stack([s[k] for s in seq], axis=0)
    return stacked

    stacked = {k: np.stack([s[k] for s in seq], axis=0) for k in seq[0].keys()}
    return stacked


def spec(thing: dict[str, np.ndarray]):
    return jax.tree.map(lambda x: x.shape, thing)


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


example = {
    "box_center": [2],
    "box_size": [1],
    "focal_length": [2],
    "img": [3, 224, 224],
    "img_wrist": [3, 224, 224],
    "img_size": [2],
    "personid": [1],
    "cam": [3],
    "cam_t": [3],
    "cam_t_full": [3],
    "keypoints_2d": [21, 2],
    "keypoints_3d": [21, 3],
    "mano.betas": [10],
    "mano.global_orient": [1, 3, 3],
    "mano.hand_pose": [15, 3, 3],
    "vertices": [778, 3],
    "right": [1],
    "scaled_focal_length": [1],
}
example = remap_keys(example)


def postprocess_sequence(seq: list[dict]):
    """Filters frames to ensure only right-hand detections are included.
    returns:
        dict: the filtered seq
        bool: whether the seq is usable
    """

    is_right = np.array([s["right"].mean() for s in seq]).mean()
    is_right = int(is_right > 0.5)

    if not is_right:  # skip if predominantly left hand
        return {}, False

    def select_hand(s):
        # print("select...")

        def overbatched(k):
            return (len(example[k]) + 1) < len(s[k].shape)

        while any(overbatched(k) for k in s.keys()):
            ob = {k: overbatched(k) for k in s.keys()}
            for k in s.keys():
                if ob[k]:
                    s[k] = s[k].squeeze()

            # print("Overbatched?", {k: overbatched(k) for k in s.keys()})

        def _select(s, k):
            if not len(s[k].shape):  # item only... scaled_focal_length
                return s[k]
            if s[k].shape[0] == 1:
                return s[k][0]
            return s[k][is_right]

        return {k: _select(s, k) for k in s.keys()}

    # print("Selecting hand...")
    out = [select_hand(s) for s in seq]
    return out, True


def solve_2d(out: dict):
    """solves for 2d points from 3d points and perspective matrix
    the original 2d points are wrt the box, not the full image

    Args:
        out (dict): the output from the model
        FOR ONE FRAME
    """

    frame = out["img"]
    f = out["scaled_focal_length"]
    P = perspective_projection(f, H=frame.shape[0], W=frame.shape[1])

    points = out["keypoints_3d"][None]  # expected to be batched
    size = 224  # Final crop size

    transform = center_crop(size=size, seed=None, img=frame)
    frame = apply_uv(frame, mat=transform, dsize=(size, size))
    T = solve_uv2xyz(points, P=P, U=transform)

    points3d = apply_xyz(points, mat=T)
    points2d = apply_persp(points3d, P)

    # we are operating on one frame so unbatch the points
    out["keypoints_3d"] = points3d[0]
    out["keypoints_2d"] = remove_col(points2d[0])
    out["img"] = frame
    return out


def main():

    hamer = HamerController("aisec-102.cs.luc.edu", 8001)
    # hamer = HamerController("0.0.0.0", 8001)
    files = list(MANO_1.glob("*.npz"))

    for path in tqdm(files, leave=False):
        print(f"Processing video: {path.name}")

        data = np.load(path)
        data = {k: data[k] for k in data.files}

        for k, v in tqdm(data.items(), leave=False):  # for each view in the episode
            outs = []
            for i, frame in tqdm(enumerate(v), total=len(v)):

                out = hamer(frame)
                try:
                    # patch because img is the only one with no batch dim
                    out["img"] = out["img"][None]
                except: # hamer failed ie: person not in frame
                    continue

                out = remap_keys(out)
                if out is None or out == {}:
                    continue

                outs.append(out)

            if len(outs) == 0: # hamer failed for all frames?
                continue
            outs, ok = postprocess_sequence(outs)
            if not ok:
                continue

            # list of dict to dict of stacked np arrays
            outs = stack(outs)
            outs = select_keys(outs)

            # viz
            for i, frame in enumerate(outs["img"]):
                step = jax.tree.map(lambda x: x[i], outs)
                step = solve_2d(step)

                bgr2rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                step['img'] = bgr2rgb(step['img'])
                step['img_wrist'] = bgr2rgb(step['img_wrist'])

                for k, v in step.items():
                    outs[k][i] = v  # save back to the original dict

                # pprint(spec(step))

            """
            for i, frame in enumerate(outs["img"]):
                step = jax.tree.map(lambda x: x[i], outs)
                img = render_openpose(frame, add_col(step["keypoints_2d"]))
                cv2.imshow("frame", img)
                cv2.waitKey(100)
            cv2.waitKey(0)
            """

            np.savez(MANO_4 / f"{path.stem}_{k}_filtered.npz", **outs)

        # shutil.move(str(path), MANO_1DONE / path.name)


if __name__ == "__main__":
    main()
