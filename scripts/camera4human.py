import time

import cv2
import imageio
import numpy as np

from xgym.utils import camera as cu


def main():

    cams = cu.list_cameras()

    for k, cam in cams.items():
        print(f"Camera {k}: {cam.get(cv2.CAP_PROP_FPS)} FPS")

    fps = 30
    dt = 1 / fps

    print(cams)
    frames = {k: [] for k in cams.keys()}
    i = 0
    while True:

        tic = time.time()

        _ = {k: cam.grab() for k, cam in cams.items()}
        imgs = {k: cam.retrieve()[1] for k,cam in cams.items()}
        imgs = {k: cu.square(f) for k, f in imgs.items()}
        imgs = {k: cv2.resize(f, (244, 244)) for k, f in imgs.items()}

        for k, v in imgs.items():
            frames[k].append(v)

        # if i % 5 == 0:
        print(f"showing frame {i}")
        all_imgs = np.concatenate(list(imgs.values()), axis=1)
        cv2.imshow("frame", all_imgs)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        i += 1
        toc = time.time()
        elapsed = toc - tic
        time.sleep(max(0, dt - elapsed))

    for k, v in frames.items():
        cu.save_frames(v, f"test_{k}", ext="mp4", fps=30)

    quit()


if __name__ == "__main__":
    main()
