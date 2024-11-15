import os.path as osp
from collections import OrderedDict
from pprint import pprint

import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

import local
import util
from transforms import (center_crop, random_resized_crop, random_rot,
                        random_xflip)
from util import (add_col, apply_persp, apply_uv, apply_xyz, remove_col,
                  solve_uv2xyz)


def perspective_projection(focal_length, H, W):
    """computes perspective projection of 3D points"""
    f = focal_length
    P = np.array(
        [
            [f, 0, W / 2, 0],
            [0, f, H / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ]
    )
    return P


def pipe2(img, out):
    f = out["scaled_focal_length"]
    P = perspective_projection(f, H=img.shape[0], W=img.shape[1])

    points = out["pred_keypoints_3d"] + out["pred_cam_t_full"][:, None]

    size = None
    transforms = OrderedDict(
        center_crop={
            "func": center_crop,
            "kwargs": {"size": (size := 224)},
            "rules": {"dsize": (size, size)},
        },
        center_crop2={
            "func": center_crop,
            "kwargs": {"size": (size := 600)},
            "rules": {"dsize": (size, size)},
        },
        random_resized_crop={
            "func": random_resized_crop,
            "kwargs": {
                "scale": [0.8, 1.2],
                "ratio": [0.8, 1.2],
                "tx": [-0.1, 0.1],
                "ty": [-0.1, 0.1],
            },
            "rules": {"dsize": (size, size)},
        },
        random_rot={
            "func": random_rot,
            "kwargs": {"deg": [-22.5, 22.5]},
            "rules": {"dsize": (size, size)},
        },
        random_xflip={
            "func": random_xflip,
            "kwargs": {"prob": 0.5},
            "rules": {"dsize": (size, size)},
        },
        random_occlusion={
            "func": random_occlusion,  # doesnt have to occlude the hand it can be anywhere
            "kwargs": {
                "prob": 0.5,
                "area": [0.0, 0.1],
                "ratio": [0.5, 2.0],
                "nboxes": [1, 3],
            },
            "rules": {"dsize": (size, size)},
        },
    )

    prng = jax.random.PRNGKey(2)

    U = np.eye(4)
    for t, v in transforms.items():
        prng, seed = jax.random.split(prng)
        _U = v["func"](**v["kwargs"], seed=seed, img=img)

        img = apply_uv(img, mat=_U, **v["rules"])
        cv2.imshow("frame", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        U = _U @ U
        print(U)

    T = solve_uv2xyz(points, P=P, U=U)
    points = apply_xyz(points, mat=T)
    points = apply_persp(points, P)

    img = local.render_openpose(img, points[0])
    img = local.render_openpose(img, points[1])
    cv2.imshow("frame", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    quit()


def main():
    frames = []

    vpath = osp.expanduser("output.mp4")
    reader = iter(imageio.get_reader(vpath))
    reader = [next(reader) for _ in range(100)][80:]

    for i, img in tqdm(enumerate(reader), total=len(reader)):
        out = np.load(f"data/pose_{i+80}.npz", allow_pickle=True)
        out = {k: out[k] for k in out.files}

        print(i)
        pipe2(img, out)
        continue


if __name__ == "__main__":
    main()
