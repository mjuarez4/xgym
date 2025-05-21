from dataclasses import dataclass
import os
from pathlib import Path
import shutil

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm
import tyro


@dataclass
class RunCFG:
    dir: str = "."
    file: str = ""
    height: int = 224
    width: int = 224
    channels: int = 3


def main(cfg: RunCFG):
    file_path = Path(cfg.dir) / cfg.file

    # Load .npz file as dict
    ep = dict(np.load(file_path, allow_pickle=True))

    import jax

    spec = lambda arr: jax.tree.map(lambda x: x.shape, ep)
    pprint(spec(ep))

    # Get list of camera keys
    cam_keys = [k for k in ep if ep[k].ndim == 4 and ep[k].shape[-1] == 3]

    if not cam_keys:
        print("No camera keys found in the npz file.")
        return

    # Assume all cams have the same number of frames
    num_frames = ep[cam_keys[0]].shape[0]

    import cv2

    for i in tqdm(range(num_frames)):
        # Build current frame dict from all entries
        step = {
            k: v[i]
            for k, v in ep.items()
            if isinstance(v, np.ndarray) and v.ndim == 4 and v.shape[0] == num_frames
        }
        imgs = {
            k: v
            for k, v in step.items()
            if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3
        }

        imgs = np.concatenate(list(imgs.values()), axis=0)

        cv2.imshow("img", imgs)
        key = cv2.waitKey(5)
        if key == ord("q") or key == ord("Q"):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # File cleanup prompt
    key = input("keep file or no? ").strip().lower()
    npz_file = str(file_path)
    json_file = os.path.splitext(npz_file)[0] + ".json"

    if key == "y":
        dst_dir = os.path.join(cfg.dir, "good")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(npz_file, os.path.join(dst_dir, os.path.basename(npz_file)))
        if os.path.exists(json_file):
            shutil.move(json_file, os.path.join(dst_dir, os.path.basename(json_file)))
        print(f"Moved {cfg.file} and its JSON to 'good' folder.")
    elif key == "n":
        if os.path.exists(json_file):
            os.remove(json_file)
        os.remove(npz_file)
        print(f"Deleted {cfg.file}")
    else:
        print("Skipping cleanup.")

    # Optional: animate just one camera for visual inspection
    first_camera_frames = ep[cam_keys[0]]
    fig, ax = plt.subplots()
    im = ax.imshow(first_camera_frames[0], animated=True)
    ax.axis("off")

    def update(frame):
        im.set_data(first_camera_frames[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=range(first_camera_frames.shape[0]), interval=50, blit=True
    )
    print("Showing animation")
    plt.show()


if __name__ == "__main__":
    main(tyro.cli(RunCFG))
