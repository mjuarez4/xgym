from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

import xgym
from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose


@dataclass
class Reader:

    embodiment: Union["single", "mano"] = "mano"
    verbose: bool = False
    visualize: bool = True


def overlay(orig, img, opacity):
    alpha = 1 - (0.2 * opacity)
    blended = cv2.addWeighted(orig, alpha, img, 1 - alpha, 0)
    return blended


def overlay_palm(img, x, y, opacity, size=None):
    """overlay white circle at palm location"""

    # Define circle properties
    center = (x, y)
    radius = int(size/2) if size is not None else 10
    color = (255, 255, 255)
    thickness = -1

    out = img.copy()
    cv2.circle(out, center, radius, color, thickness)
    alpha = 1 - (0.2 * opacity)
    blended = cv2.addWeighted(out, alpha, img, 1 - alpha, 0)

    return blended


def main():

    cfg = Reader()

    ds = tfds.load(f"xgym_lift_{cfg.embodiment}")["train"]

    for ep in ds:
        steps = [x for x in ep["steps"]]
        for i, s in tqdm(enumerate(steps)):

            if cfg.verbose:
                print(s.keys())

            if cfg.embodiment == "single":
                action = np.array(s["action"]).tolist()
                action = np.array([round(a, 4) for a in action])
                print(action)
                img = np.concatenate(list(s["observation"]["image"].values()), axis=1)
            else:
                img = np.array(s["observation"]["frame"])
                for j in range(4):
                    try:
                        points2d = add_col(
                            np.array(steps[i + j]["observation"]["keypoints_2d"])
                        )
                        palm = points2d[0]
                        img = overlay(img, render_openpose(img, points2d), j)
                        print(palm)
                        # z = np.array(steps[i + j]["observation"]["keypoints_3d"])[0][2]
                        # img = overlay_palm(img, int(palm[0]), int(palm[1]), j, z)
                    except Exception as e:
                        print(e)

            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(50)

        print()


if __name__ == "__main__":
    main()
