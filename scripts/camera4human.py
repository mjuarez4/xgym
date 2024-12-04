import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import draccus
import imageio
import numpy as np
from tqdm import tqdm

from xgym.utils import camera as cu


@dataclass
class RunCFG:

    task: str = input("Task: ").lower()
    base_dir: str = Path("~/data").expanduser()
    time: str = time.strftime("%Y%m%d-%H%M%S")
    env_name: str = f"xgym-mano-{task}-{time}"
    data_dir: str = base_dir / env_name
    nsteps: int = 300

    def __post_init__(self):

        assert self.task, "Task must be provided"
        self.data_dir.mkdir(parents=True, exist_ok=True)


CAM_MAP = {
    0: "worm",
    2: "overhead",
    10: "side",
}


def du_flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(du_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flush(episode: dict, ep: int, cfg: RunCFG):

    episode = {CAM_MAP[k]: v for k, v in episode.items() if k in CAM_MAP}
    out = str(cfg.data_dir / f"ep{ep}")
    np.savez(out, **episode)

    # cu.save_frames(v, str(cfg.data_dir / f"ep{ep}_{k}"), ext="mp4", fps=30)


@draccus.wrap()
def main(cfg: RunCFG):

    cams = cu.list_cameras()

    for k, cam in cams.items():
        print(f"Camera {k}: {cam.get(cv2.CAP_PROP_FPS)} FPS")

    fps = 30
    dt = 1 / fps

    print(cams)
    for ep in tqdm(range(100)):
        frames = {k: [] for k in cams.keys()}
        for step in tqdm(range(cfg.nsteps), leave=False):

            tic = time.time()

            _ = {k: cam.grab() for k, cam in cams.items()}
            imgs = {k: cam.retrieve()[1] for k, cam in cams.items()}
            imgs = {k: cu.square(f) for k, f in imgs.items()}
            imgs = {k: cv2.resize(f, (224, 224)) for k, f in imgs.items()}

            for k, v in imgs.items():
                frames[k].append(v)

            # if step % 6 == 0: # 5 FPS show
            # print(f"showing frame {step}")

            all_imgs = np.concatenate(list(imgs.values()), axis=1)
            cv2.imshow("frame", all_imgs)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            toc = time.time()
            elapsed = toc - tic
            time.sleep(max(0, dt - elapsed))

        flush(frames, ep, cfg)

    quit()


if __name__ == "__main__":
    main()
