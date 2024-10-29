from typing import List

import cv2
import imageio
import numpy as np


def square(img: np.array):
    """returns a square center crop based on the smaller side"""
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y : y + s, x : x + s]


def resize_all(imgs: List[np.array]):
    m = max([f.shape[0] for f in imgs])
    return [cv2.resize(f, (m, m)) for f in imgs]


def list_cameras():
    index = 0
    arr = {}
    while index < 20:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.set(cv2.CAP_PROP_FPS, 60)
            arr[index] = cap
            # cap.release()
        index += 1
    return arr


def save_frames(frames, path, ext="mp4", fps=30):
    path = f"{path}.{ext}" if not path.endswith(f".{ext}") else path
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
