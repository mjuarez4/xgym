from dataclasses import dataclass
from pprint import pprint
from typing import Union

import cv2
import numpy as np
from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose


def overlay_img(orig, img, opacity):
    """overlay openpose pose on image with opacity factor"""
    alpha = 1 - (0.2 * opacity)
    blended = cv2.addWeighted(orig, alpha, img, 1 - alpha, 0)
    return blended


def overlay_pose(img, pose, opacity):
    """overlay openpose pose on image with opacity factor"""

    out = render_openpose(img.copy(), pose)
    blended = overlay_img(out, img, opacity)
    return blended


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def overlay_palm(img, x, y, opacity, color=WHITE, size=None):
    """overlay white circle at palm location"""

    # Define circle properties
    center = (x, y)
    radius = size if size is not None else 5
    thickness = -1

    out = img.copy()
    cv2.circle(out, center, radius, color, thickness)
    blended = overlay_img(out, img, opacity)
    return blended
