import enum
from dataclasses import dataclass
from pprint import pprint
from typing import Union

import cv2
import draccus
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import xgym
from tqdm import tqdm
from xgym.rlds.util import add_col, remove_col
from xgym.rlds.util.render import render_openpose
from xgym.rlds.util.trajectory import binarize_gripper_actions, scan_noop
from xgym.viz.mano import overlay_palm, overlay_pose


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


class Reader:
    pass


class RLDSReader:
    pass


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

    allnoops = 0
    for ep in tqdm(ds, total=len(ds)):
        steps = [x for x in ep["steps"]]

        # need them in advance to binarize
        pos = [s['observation']["proprio"]['position'] for  s in steps]
        pos = np.array(pos)
        gripper = pos[:, -1]
        gripper = binarize_gripper_actions(jnp.array(gripper), open=0.90, close=0.65)
        pos[:, -1] = np.array(gripper.tolist())

        noops = np.array(scan_noop(jnp.array(pos),threshold=1e-2))
        mask = ~noops

        pos = pos[mask]
        actions = pos[1:] - pos[:-1]
        actions[:,-1] = pos[:-1,-1] 

        ifprint(actions.round(2))
        allnoops += np.array(noops).sum()
        ifprint(f'noops: {np.array(noops).sum()}')
        ifprint(f"len: {len(steps)}")
        continue

        for i, s in tqdm(enumerate(steps), total=len(steps), leave=False):

            # if cfg.verbose:
            # ifprint(s.keys())

            if cfg.embodiment == "single":
                ifprint(actions[i].round(2))
                img = np.concatenate(list(s["observation"]["image"].values()), axis=1)
                imshow("image", recolor(img))

            else:
                try:
                    img = np.array(s["observation"]["img"])
                except KeyError:
                    img = np.array(s["observation"]["frame"])

                ifprint(steps[i]["observation"]["keypoints_3d"])
                for j in range(0, cfg.horizon, cfg.resolution):
                    try:

                        points2d = add_col(
                            np.array(steps[i + j]["observation"]["keypoints_2d"])
                        )
                        palm = points2d[0]

                        # ifprint(img.mean(0).mean(0))

                        if cfg.use_palm:
                            # z = np.array(steps[i + j]["observation"]["keypoints_3d"])[0][2]
                            img = overlay_palm(
                                img, int(palm[0]), int(palm[1]), opacity=j
                            )
                        else:
                            img = overlay_pose(img, points2d, opacity=j)
                        # ifprint(palm)

                        wrist = np.array(s["observation"]["img_wrist"])
                        imshow("wrist", cv2.resize(recolor(wrist), (640, 640)))

                    except (IndexError, KeyError) as e:
                        pass
                        # ifprint(e)
                imshow("image", cv2.resize(recolor(img), (640, 640)))

            if cfg.visualize:
                cv2.waitKey(cfg.wait)

        print()
    ifprint(allnoops)


if __name__ == "__main__":
    main()
