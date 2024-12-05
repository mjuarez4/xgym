from dataclasses import dataclass
from pprint import pprint
import draccus
from typing import Union

import cv2
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

import xgym
from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose


import enum

class Task(enum.Enum):
    LIFT = "lift"
    DUCK = "duck"

class Embodiment(enum.Enum):
    SINGLE = "single"
    MANO = "mano"

@dataclass
class BaseReaderConfig:

    embodiment: Embodiment = Embodiment.SINGLE
    task: Task = Task.LIFT
    verbose: bool = False
    visualize: bool = True

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, enum.Enum):
                setattr(self, k, v.value)

def overlay(orig, img, opacity):
    alpha = 1 - (0.2 * opacity)
    blended = cv2.addWeighted(orig, alpha, img, 1 - alpha, 0)
    return blended


def overlay_palm(img, x, y, opacity, size=None):
    """overlay white circle at palm location"""

    # Define circle properties
    center = (x, y)
    radius = int(size / 2) if size is not None else 10
    color = (255, 255, 255)
    thickness = -1

    out = img.copy()
    cv2.circle(out, center, radius, color, thickness)
    alpha = 1 - (0.2 * opacity)
    blended = cv2.addWeighted(out, alpha, img, 1 - alpha, 0)

    return blended


@draccus.wrap()
def main(cfg: BaseReaderConfig):
    pprint(cfg)

    name = f"xgym_{cfg.task}_{cfg.embodiment}"
    print(f"Loading {name} dataset")
    ds = tfds.load(name)["train"]
    # ds = ds.repeat(100)

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
                img = np.array(s["observation"]["img"])
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

                    except IndexError as e:
                        pass
                        # print(e)

                wrist = np.array(s["observation"]["img_wrist"])
                cv2.imshow("wrist", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))

            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(50)

        print()


if __name__ == "__main__":
    main()
