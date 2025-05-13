from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("WebAgg")  # Use TkAgg backend for interactive plotting
import cv2
import jax
import numpy as np
from matplotlib import pyplot as plt
from rich.pretty import pprint
from tqdm import tqdm

from april import Detector, process_frame

# numpy no scientific notation
np.set_printoptions(suppress=True, precision=3)


def main():
    cal = list(Path("calibrate").rglob("*.dat"))
    pprint(cal)

    info, dat = read(cal[-1])

    spec = lambda arr: jax.tree.map(
        lambda x: x.shape if isinstance(x, np.ndarray) else type(x), arr
    )
    pprint(spec(dat))

    # Default intrinsics (approximate for 720p MacBook webcam)
    fx = 600.0  # focal length in pixels
    fy = 600.0
    cx = 640 / 2  # 320.0
    cy = 480 / 2  # 240.0
    camera_params = (fx, fy, cx, cy)
    tag_size = 0.0762  # meters (3 inches)

    # Create AprilTag detector with your specified parameters
    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    trajectory = []
    poses = []
    centers = []
    for i in tqdm(range(len(dat["time"][:100:3]))):

        d = jax.tree.map(lambda x: x[i], dat)
        frame = d["/xgym/camera/low"]
        rbt = d["xarm_pose"]
        poses.append(rbt)
        size = frame.shape
        scale = 4
        frame = cv2.resize(frame, (size[1] * scale, size[0] * scale))
        # pprint(spec(d))
        pos, center = process_frame(frame)
        trajectory.append(pos if pos is not None else trajectory[-1])

        if center is not None:
            centers.append(center)
        for c in centers:
            cv2.circle(frame, c, 5, (0, 0, 255), -1)

        cv2.imshow("AprilTag Detection with Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # print(d["camera"]["pose"])
        # print(d["camera"]["pose"]["rotation"])
        # print(d["camera"]["pose"]["translation"])

        # print(d["camera"]["intrinsics"])
        # print(d["camera"]["intrinsics"]["fx"])
        # print(d["camera"]["intrinsics"]["fy"])
        # print(d["camera"]["intrinsics"]["cx"])
        # print(d["camera"]["intrinsics"]["cy"])

    cv2.destroyAllWindows()

    trajectory = np.array(trajectory)
    poses = np.array(poses)
    poses[:, :3] /= 1e3
    pprint(trajectory)
    pprint(poses)
    quit()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "3d"})
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Tag Trajectory"
    )
    ax.scatter(
        trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c="g", label="Start"
    )
    ax.scatter(
        trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c="r", label="End"
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_title("Estimated Tag Trajectory in Camera Frame")
    plt.tight_layout()
    # plt.savefig("trajectory.png")
    plt.show()

    # cv2.calibrateCamera
    # cv2.calibrateHandEye
    # cv2.calibrateRobotWorldHandEye


if __name__ == "__main__":
    main()
