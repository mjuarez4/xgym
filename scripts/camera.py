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
    frames = []
    while True:

        tic = time.time()

        _ = [cam.grab() for cam in cams.values()]
        imgs = [im for (ret, im) in [cam.retrieve() for cam in cams.values()]]
        imgs = [cu.square(f) for f in imgs]

        # resize by the biggest dimension of frames
        imgs = cu.resize_all(imgs)
        frame = np.concatenate(imgs, axis=1)

        frames.append(frame)
        cv2.putText(
            frame,
            f"FPS: 5 | time: {tic}",
            (512, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        if len(frames) % 5 == 0:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        toc = time.time()
        elapsed = toc - tic
        time.sleep(max(0, dt - elapsed))

    cu.save_frames(frames, "test", ext="mp4", fps=30)

    quit()


# frames = np.array(frames)


if __name__ == "__main__":
    main()
