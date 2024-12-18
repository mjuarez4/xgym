from dataclasses import dataclass
import xgym
from pprint import pprint
import draccus
from typing import Union

import cv2
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose

from xgym.viz.mano import overlay_pose, overlay_palm

import enum

class Task(enum.Enum):
    LIFT = "lift"
    STACK = "stack"
    DUCK = "duck"
    POUR = "pour"

class Embodiment(enum.Enum):
    SINGLE = "single"
    MANO = "mano"

@dataclass
class BaseReaderConfig:

    embodiment: Embodiment = Embodiment.SINGLE
    task: Task = Task.LIFT
    version: str = ""
    verbose: bool = False
    visualize: bool = True

    use_palm: bool = False
    wait: int = 50
    horizon: int = 10
    resolution: int = 3

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, enum.Enum):
                setattr(self, k, v.value)


def recolor(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

@draccus.wrap()
def main(cfg: BaseReaderConfig):
    pprint(cfg)

    name = f"xgym_{cfg.task}_{cfg.embodiment}{cfg.version}"
    xgym.logger.info(f"Loading {name} dataset")
    ds = tfds.load(name)["train"]
    xgym.logger.info(len(ds))

    # ds = ds.repeat(100)

    def imshow(*args):
        if cfg.visualize:
            cv2.imshow(*args)

    def ifprint(x):
        if cfg.verbose:
            print(x)


    for ep in tqdm(ds,total=len(ds)):
        steps = [x for x in ep["steps"]]

        for i, s in tqdm(enumerate(steps), total=len(steps), leave=False):

            if cfg.verbose:
                ifprint(s.keys())

            if cfg.embodiment == "single":
                action = np.array(s["action"]).tolist()
                action = np.array([round(a, 4) for a in action])
                ifprint(action)
                img = np.concatenate(list(s["observation"]["image"].values()), axis=1)
                cv2.imshow("image", recolor(img))
            else:
                try:
                    img = np.array(s["observation"]["img"])
                except KeyError:
                    img = np.array(s["observation"]["frame"])

                ifprint(steps[i]['observation']['keypoints_3d'])
                for j in range(0,cfg.horizon,cfg.resolution):
                    try:

                        points2d = add_col(
                            np.array(steps[i + j]["observation"]["keypoints_2d"])
                        )
                        palm = points2d[0]

                        ifprint(img.mean(0).mean(0))

                        if cfg.use_palm:
                            # z = np.array(steps[i + j]["observation"]["keypoints_3d"])[0][2]
                            img = overlay_palm(img, int(palm[0]), int(palm[1]), opacity=j)
                        else:
                            img = overlay_pose(img, points2d, opacity=j)
                        # ifprint(palm)

                        wrist = np.array(s["observation"]["img_wrist"])
                        cv2.imshow("wrist", cv2.resize(recolor(wrist), (640, 640)))

                    except (IndexError, KeyError) as e:
                        pass
                        # ifprint(e)
                cv2.imshow("image", cv2.resize(recolor(img), (640, 640)))

            cv2.waitKey(cfg.wait)

        print()


if __name__ == "__main__":
    main()
