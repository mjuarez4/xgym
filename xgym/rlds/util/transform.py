import os.path as osp
from collections import OrderedDict
from pprint import pprint

import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ._util import add_col, remove_col


def random_resized_crop(
    seed, scale=[0.8, 1.2], ratio=[0.8, 1.2], tx=[-0.1, 0.1], ty=[-0.1, 0.1], **kwargs
):
    """generates random resized crop matrix for image and keypoints"""

    seed, *rng = jax.random.split(seed, 5)

    z = jax.random.uniform(rng[0], shape=(), minval=scale[0], maxval=scale[1])
    tx = jax.random.uniform(rng[1], shape=(), minval=tx[0], maxval=tx[1])
    ty = jax.random.uniform(rng[2], shape=(), minval=ty[0], maxval=ty[1])

    # Generate random ratio within specified range
    log_ratio = jax.random.uniform(
        rng[3], shape=(), minval=np.log(ratio[0]), maxval=np.log(ratio[1])
    )
    aspect = jnp.exp(log_ratio)  # Convert log scale back to linear scale

    mat_uv = np.array(
        [
            [aspect, 0, -tx, 0],
            [0, 1 / aspect, -ty, 0],
            [0, 0, 1 / z, 0],
            [0, 0, 0, 1],
        ]
    )

    return mat_uv


def center_crop(img, size=224, seed=None, **kwargs):
    """generates center crop matrix for image and keypoints"""

    H, W = img.shape[:2]
    side = min(H, W)
    x1, x2, y1, y2 = (W - side) // 2, (W + side) // 2, (H - side) // 2, (H + side) // 2
    z = size / side
    # print(z, size, side)

    mat_uv = np.array(
        [
            [1, 0, -x1, 0],
            [0, 1, -y1, 0],
            [0, 0, 1 / z, 0],
            [0, 0, 0, 1],
        ]
    )
    return mat_uv


def random_xflip(seed, prob=0.5, **kwargs):
    k = jax.random.uniform(seed, shape=(), minval=0, maxval=1)
    k = k < prob
    sx = -1 if k else 1
    H, W = kwargs["img"].shape[:2]

    mat_uv = np.array(
        [
            # [1 - 2 * k, 0, k * kwargs["img"].shape[0], 0],
            [sx, 0, (W * (sx == -1)), 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).astype(np.float32)
    return mat_uv


def random_rot(seed, deg=[-22.5, 22.5], **kwargs):
    # Generate random rotation angle in radians
    k = jax.random.uniform(seed, shape=(), minval=deg[0], maxval=deg[1])
    k = (k / 180) * jnp.pi
    H, W = kwargs["img"].shape[:2]

    # Compute cos and sin of the angle
    cos = jnp.cos(k)
    sin = jnp.sin(k)

    t = np.array(
        [
            [1, 0, -W / 2, 0],
            [0, 1, -H / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    rotate = np.array(
        [
            [cos, -sin, 0, 0],
            [sin, cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    tinv = np.linalg.inv(t)
    mat_uv = tinv @ rotate @ t
    return mat_uv
