import cv2
import imageio
import numpy as np

from xgym.utils import camera as cu


def main():

    cams = cu.list_cameras()

    frames = []
    while True:

        imgs = [im for (ret, im) in [cam.read() for cam in cams.values()]]
        imgs = [cu.square(f) for f in imgs]

        # resize by the biggest dimension of frames
        imgs = cu.resize_all(imgs)

        frame = np.concatenate(imgs, axis=1)

        frames.append(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cu.save_frames(frames, "test", ext="mp4", fps=30)

    quit()

# frames = np.array(frames)


if __name__ == "__main__":
    main()
