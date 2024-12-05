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
    "box_center": [2],
    "box_size": [1],
    "focal_length": [2],
    "img": [3, 256, 256],
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
    pprint(spec(out))
    return out, True


def select_keys(out: dict):

    out = {
        k.replace("pred_", "").replace("_params", ""): v.squeeze()
        for k, v in out.items()
    }
    out["keypoints_3d"] += out.pop("cam_t_full")[None, :]

    keep = [
        # "box_center",
        # "box_size",
        "img",
        # "img_size",
        # "personid",
        # "cam",
        # "cam_t",
        # "cam_t_full", # used above
        # "keypoints_2d", # these are wrt the box not full img
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


def solve_2d(frame, out):
    out = out.data if hasattr(out, "data") else out
    f = out["scaled_focal_length"]
    print("persp start")
    P = perspective_projection(f, H=frame.shape[0], W=frame.shape[1])
    print("persp end")
    points = out["keypoints_3d"]
    size = 224  # Final crop size

    # Apply center cropping
    transform = center_crop(size=size, seed=None, img=frame)
    print("uv start")
    frame = apply_uv(frame, mat=transform, dsize=(size, size))
    print("uv end")
    print("solve start")
    T = solve_uv2xyz(points, P=P, U=transform)
    print("solve end")

    print("apply xyz start")
    print(f"points shape: {points.shape}, T shape: {T.shape}")
    points3d = apply_xyz(points, mat=T)
    print("apply xyz end")
    print("apply persp start")
    points2d = apply_persp(points3d, P)
    print("apply persp end")

    out["keypoints_3d"] = points3d
    out["keypoints_2d"] = points2d
    out["img"] = frame
    return out


def main():

    hamer = HamerController("aisec-102.cs.luc.edu", 8001)

    files = list(MANO_1.glob("*.npz"))
    print(files)

    # files = [f for f in files if "7" in f.name]

    for path in tqdm(files, leave=False):
        print(f"Processing video: {path.name}")

        data = np.load(path)
        data = {k: data[k] for k in data.files}

        for k, v in tqdm(data.items(), leave=False):  # for each view in the episode
            outs = []
            for i, frame in tqdm(enumerate(v), total=len(v)):

                cv2.imshow("frame", frame)
                cv2.waitKey(1)

                out = hamer(frame)
                out = select_keys(out)
                # {k: v.squeeze() for k, v in out.items()}
                pprint(spec(out))
                if out is None or out == {}:
                    continue
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
