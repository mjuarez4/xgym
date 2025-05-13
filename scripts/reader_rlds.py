import enum
from dataclasses import dataclass
from pprint import pprint
from typing import Union

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import tyro
from jax import numpy as jnp
from tqdm import tqdm

import xgym
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

    version: str  # (e.g. "4.0.3")
    embodiment: Embodiment = Embodiment.SINGLE
    task: Task = Task.LIFT

    verbose: bool = False
    visualize: bool = True

    show_image: bool = True  # replacement for visualize
    show_summary: bool = False

    use_palm: bool = False
    wait: int = 20  # the cv2 wait time in ms
    horizon: int = 10
    resolution: int = 3

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, enum.Enum):
                setattr(self, k, v.value)


def recolor(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class Reader:

    def __init__(self, cfg: BaseReaderConfig):
        self.cfg = cfg

    pass

    def imshow(self, images):
        if self.cfg.visualize:
            img = np.concatenate(list(images), axis=1)
            cv2.imshow("image", recolor(img))
            cv2.waitKey(self.cfg.wait)

    def print(self, *args, **kwargs):
        if self.cfg.verbose:
            print(*args, **kwargs)


class RLDSReader(Reader):

    def __call__(self):
        pass

    def read(self, step):

        if self.cfg.embodiment == "single":
            self.print(step["action"])
            self.imshow(list(step["observation"]["image"].values()))

        if self.cfg.embodiment == "mano":
            raise NotImplementedError

            img = np.array(step["observation"]["img"], step["observation"]["frame"])

            if not self.cfg.visualize:
                return

            self.print(step["observation"]["keypoints_3d"])
            for j in range(0, self.cfg.horizon, self.cfg.resolution):
                try:

                    points2d = add_col(
                        np.array(steps[i + j]["observation"]["keypoints_2d"])
                    )
                    palm = points2d[0]

                    # ifprint(img.mean(0).mean(0))

                    if cfg.use_palm:
                        # z = np.array(steps[i + j]["observation"]["keypoints_3d"])[0][2]
                        img = overlay_palm(img, int(palm[0]), int(palm[1]), opacity=j)
                    else:
                        img = overlay_pose(img, points2d, opacity=j)
                    # ifprint(palm)

                    wrist = np.array(s["observation"]["img_wrist"])
                    imshow("wrist", cv2.resize(recolor(wrist), (640, 640)))

                except (IndexError, KeyError) as e:
                    pass
                    # ifprint(e)
            imshow("image", cv2.resize(recolor(img), (640, 640)))


def summary():

    fig, axs = plt.subplots(14, 1, figsize=(20, 40))

    ds = ds.take(100)

    reader = RLDSReader(cfg)
    for ep in tqdm(ds, total=len(ds)):
        steps = [x for x in ep["steps"]]
        # for i, s in tqdm(enumerate(steps), total=len(steps), leave=False):

        joints = [s["observation"]["proprio"]["joints"] for s in steps]
        grip = [s["observation"]["proprio"]["gripper"] for s in steps]

        selected = np.concatenate([joints, grip], axis=1)
        ### filter joints noop
        noops = np.array(scan_noop(jnp.array(selected), threshold=1e-2))
        mask = ~noops
        # ep = jax.tree.map(select := lambda x: x[mask], ep)

        print(mask)
        # steps = [s for s, m in zip(steps, mask) if m]
        # quit()

        # combine list[dict[str, np.array]] into dict[str, stacked np.array]
        outs = {}
        outs2 = {}

        # for k in steps[0]["action"].keys():
        # outs[k] = np.stack([s["action"][k] for s in steps])
        # outs2[k] = np.stack([s["action"][k] for s in steps])
        # for k in steps[0]['observation']['proprio'].keys():
        # outs[k] = np.stack([s['observation']['proprio'][k] for s in steps])
        # outs2[k] = np.stack([s['observation']['proprio'][k] for s in steps])

        outs = np.concatenate(list(outs.values()), axis=1)
        outs2 = np.concatenate(list(outs2.values()), axis=1)
        outs2 = outs2[mask]

        for i in range(outs.shape[1]):
            # ax[i].plot(outs[:,i],color=(0,0,1,0.05))
            axs[i].plot(outs[:, i], color=(0, 0, 1, 1))
            axs[i].plot(outs2[:, i], color=(0, 0, 1, 1))
            axs[i].plot(np.zeros_like(outs[:, i]), color=(0, 0, 0, 0.5), linestyle="--")

        plt.pause(2)
        # clear all the axes
        for ax in axs:
            ax.clear()


def main(cfg: BaseReaderConfig):
    pprint(cfg)

    name = f"xgym_{cfg.task}_{cfg.embodiment}:{cfg.version}"
    xgym.logger.info(f"Loading {name} dataset")
    ds = tfds.load(name)["train"]
    xgym.logger.info(len(ds))

    # ds = ds.repeat(100)

    def ifprint(*args, **kwargs):
        if cfg.verbose:
            print(*args, **kwargs)

    def ifshow(*args, **kwargs):
        if cfg.visualize:
            cv2.imshow(*args, **kwargs)

    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(14, 1, figsize=(20, 40))

    ds = ds.take(10)

    reader = RLDSReader(cfg)
    for ep in tqdm(ds, total=len(ds)):
        steps = [x for x in ep["steps"]]
        # for i, s in tqdm(enumerate(steps), total=len(steps), leave=False):

        joints = [s["observation"]["proprio"]["joints"] for s in steps]
        grip = [s["observation"]["proprio"]["gripper"] for s in steps]

        selected = np.concatenate([joints, grip], axis=1)
        ### filter joints noop
        noops = np.array(scan_noop(jnp.array(selected), threshold=1e-2))
        mask = ~noops
        # ep = jax.tree.map(select := lambda x: x[mask], ep)

        # print(mask)
        # steps = [s for s, m in zip(steps, mask) if m]
        # quit()

        # combine list[dict[str, np.array]] into dict[str, stacked np.array]
        outs = {}
        outs2 = {}

        for k in steps[0]["action"].keys():
            outs[k] = np.stack([s["action"][k] for s in steps])
        # outs2[k] = np.stack([s["action"][k] for s in steps])
        # for k in steps[0]['observation']['proprio'].keys():
        # outs[k] = np.stack([s['observation']['proprio'][k] for s in steps])
        # outs2[k] = np.stack([s['observation']['proprio'][k] for s in steps])

        outs = np.concatenate(list(outs.values()), axis=1)
        # outs2 = np.concatenate(list(outs2.values()), axis=1)
        # outs2 = outs2[mask]

        for i in range(outs.shape[1]):
            # ax[i].plot(outs[:,i],color=(0,0,1,0.05))
            # axs[i].plot(outs[:, i], color=(0, 0, 1, 0.05))
            axs[i].plot(outs[:, i], color=(0, 0, 1, 1))
            axs[i].plot(np.zeros_like(outs[:, i]), color=(0, 0, 0, 0.5), linestyle="--")

        # plt.pause(2)
        # clear all the axes
        # for ax in axs:
        # ax.clear()

    plt.show()
    quit()
    # plt.plot

    # reader.read(s)

    quit()
    """

    allnoops = 0
    for ep in tqdm(ds, total=len(ds)):
        steps = [x for x in ep["steps"]]

        """
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
        """

        for i, s in tqdm(enumerate(steps), total=len(steps), leave=False):

            # if cfg.verbose:
            # ifprint(s.keys())

            if cfg.embodiment == "single":
                ifprint(actions[i].round(2))
                img = np.concatenate(list(s["observation"]["image"].values()), axis=1)
                ifshow("image", recolor(img))

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
                        ifshow("wrist", cv2.resize(recolor(wrist), (640, 640)))

                    except (IndexError, KeyError) as e:
                        pass
                        # ifprint(e)
                ifshow("image", cv2.resize(recolor(img), (640, 640)))

            if cfg.visualize:
                key = cv2.waitKey(cfg.wait)
                if key == ord("q"):
                    break

        print()
    ifprint(allnoops)


if __name__ == "__main__":
    main(tyro.cli(BaseReaderConfig))
