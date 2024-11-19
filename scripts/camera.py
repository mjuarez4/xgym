import time
from pprint import pprint

import cv2
import imageio
import numpy as np
import pyudev

from xgym.utils import camera as cu

context = pyudev.Context()

import pandas as pd

df = []
for device in context.list_devices(subsystem="video4linux"):
    df.append({
        "path": device.device_node,
        "serial": device.get("ID_SERIAL_SHORT"),
        "model": device.get("ID_MODEL"),
    })

df = pd.DataFrame(df)
# sort by int(path.split("/")[-1].replace("video", ""))
df = df.sort_values(by="path", key=lambda x: x.str.split("/").str[-1].str.replace("video", "").astype(int))

print(df)


def main():

    cams = cu.list_cameras()

    for k, cam in cams.items():
        print(f"Camera {k}: {cam.get(cv2.CAP_PROP_FPS)} FPS")

    fps = 30
    dt = 1 / fps

    # pop all but one cam

    print(cams)
    frames = []
    while True:

        tic = time.time()

        _ = [cam.grab() for cam in cams.values()]
        imgs = {
            k: im
            for k, (ret, im) in {k: cam.retrieve() for k, cam in cams.items()}.items()
        }
        imgs = {k: cu.square(f) for k, f in imgs.items()}
        imgs = cu.writekeys(imgs)

        # resize by the biggest dimension of frames
        imgs = cu.resize_all(list(imgs.values()))
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
