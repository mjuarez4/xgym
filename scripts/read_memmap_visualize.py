from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm
from rich.pretty import pprint
from pathlib import Path
import os
import shutil
import draccus
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass
class RunCFG:
    dir: str = "."
    file: str = ""
    num_frames: int = 100
    height: int = 224
    width: int = 224
    channels: int = 3


def main(cfg: RunCFG):

    file_path = Path(cfg.dir) / cfg.file

    mm = np.memmap(
        file_path,
        dtype="uint8",
        mode="r",
        shape=(cfg.num_frames, cfg.height, cfg.width, cfg.channels),
    )

    from xgym.viz.memmap import read

    info, ep = read(file_path)

    import jax

    spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)
    pprint(spec(ep))

    for i in tqdm(range(len(ep['time']))):
        step = jax.tree.map(lambda x: x[i], ep)

        imgs = {k:v for k, v in step.items() if 'cam' in k}
        imgs = np.concatenate(list(imgs.values()), axis=0)
        # show with cv2
        import cv2
        cv2.imshow("img", imgs)
        key = cv2.waitKey(5)

        if key==ord("q") or key == ord("Q"):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    key = input("keep file or no?").strip().lower()
    dat_file = str(file_path)
    json_file = os.path.splitext(dat_file)[0] + ".json"

    if key == "y":
        dst_dir = os.path.join(cfg.dir, "good")
        os.makedirs(dst_dir, exist_ok=True)
        dst_dat = os.path.join(dst_dir, os.path.basename(dat_file))
        shutil.move(dat_file, dst_dat)
        dst_json = os.path.join(dst_dir, os.path.basename(json_file))
        shutil.move(json_file, dst_json)
        print(f"moved {cfg.file} files to good folder")
    elif key == "n":
        json_file = os.path.splitext(cfg.file)[0] + ".json"
        json_file_path = os.path.join(cfg.dir, json_file)
        print(json_file_path)
        os.remove(json_file_path)
        dat_file_path = os.path.join(cfg.dir, cfg.file)
        os.remove(dat_file_path)
        print(f"deleted {dat_file_path} file")
    else:
        print("skipping")

    quit()


    first_camera_frames = mm[::2]
    fig, ax = plt.subplots()
    im = ax.imshow(first_camera_frames[0], animated=True)
    ax.axis("off")

    def update(frame):
        im.set_data(first_camera_frames[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=range(first_camera_frames.shape[0]), interval=50, blit=True
    )
    print("showing")
    plt.show()


if __name__ == "__main__":
    main(tyro.cli(RunCFG))
