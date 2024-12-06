import hamer
import imageio
import numpy as np


def pipeline(frame, out):

    # center crop to 224
    pass


def main():

    # Load the video
    # model = hamer

    video = imageio.get_reader("video.mp4")
    paths = [f"pose_{i}.npz" for i in len(video)]
    for i, (frame, path) in enumerate(zip(video, paths)):
        out = np.load(path)
        out = pipeline(frame, out)
        np.save(path, out)


if __name__ == "__main__":
    main()
