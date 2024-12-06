from dataclasses import dataclass
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
    DUCK = "duck"

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

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, enum.Enum):
                setattr(self, k, v.value)


@draccus.wrap()
def main(cfg: BaseReaderConfig):
    pprint(cfg)

    name = f"xgym_{cfg.task}_{cfg.embodiment}{cfg.version}"
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
                try:
                    img = np.array(s["observation"]["img"])
                except KeyError:
                    img = np.array(s["observation"]["frame"])

                for j in range(4):
                    try:

                        points2d = add_col(
                            np.array(steps[i + j]["observation"]["keypoints_2d"])
                        )
                        palm = points2d[0]

                        if cfg.use_palm:
                            # z = np.array(steps[i + j]["observation"]["keypoints_3d"])[0][2]
                            img = overlay_palm(img, int(palm[0]), int(palm[1]), opacity=j)
                        else:
                            img = overlay_pose(img, points2d, opacity=j)
                        print(palm)

                        wrist = np.array(s["observation"]["img_wrist"])
                        cv2.imshow("wrist", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))

                    except (IndexError, KeyError) as e:
                        pass
                        # print(e)


            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(50)

        print()


if __name__ == "__main__":
    main()
