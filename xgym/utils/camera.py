import os
from typing import Dict, List, Union

import cv2
import imageio
import numpy as np


def square(img: np.array):
    """returns a square center crop based on the smaller side"""
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y : y + s, x : x + s]


def resize_all(imgs: List[np.array], m=None):
    m = max([f.shape[0] for f in imgs]) if m is None else m
    return [cv2.resize(f, (m, m)) for f in imgs]


def list_cameras():

    cams = {}
    videos = [x for x in os.listdir("/dev/") if "video" in x]

    for i in range(len(videos) + 4):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cap.set(cv2.CAP_PROP_FPS, 60)
            cams[i] = cap
    return cams


def save_frames(frames, path, ext="mp4", fps=30):
    path = f"{path}.{ext}" if not path.endswith(f".{ext}") else path
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def tile(imgs: Union[Dict, List]):
    if isinstance(imgs, dict):
        return np.concatenate(list(imgs.values()), axis=1)
    if isinstance(imgs, list):
        return np.concatenate(imgs, axis=1)


def writekeys(imgs: Dict):
    return {
        k: cv2.putText(
            imgs[k],
            f"{k}",
            (0, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )
        for k in imgs
    }
