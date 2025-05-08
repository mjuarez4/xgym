import time
from pprint import pprint

import cv2
import imageio
import numpy as np
import tyro
from xgym import BASE
from xgym.calibrate.april import Calibrator
from xgym.calibrate.urdf.robot import RobotTree
from xgym.utils import camera as cu

from get_calibration import SPEED_JOINTS, Config, MyCamera, Xarm


def main(cfg: Config):

    xarm = Xarm(cfg.ip)
    xarm.speed = SPEED_JOINTS

    urdf = RobotTree()
    cal = Calibrator()

    cam = MyCamera(0)

    cam2base = np.load(BASE / "base.npz")["arr_0"]

    while True:

        frame = cam.read()
        joints = xarm.angles

        kins = urdf.set_pose(joints)
        kins = {k: v.get_matrix().numpy()[0] for k, v in urdf.set_pose(joints).items()}
        keypoints = {k: cam2base @ v for k, v in kins.items()}

        cal.draw_pose_axes(
            frame,
            cam2base[:3, :3],
            cam2base[:3, 3],
        )

        points = []
        for k, v in keypoints.items():
            point = cal.project_points(
                (0, 0, 0),
                v[:3, :3],
                v[:3, 3],
            )
            points.append(point.reshape(-1))
        for a, b in zip(points[:-1], points[1:]):
            # print(a.shape, b.shape)
            pink = (255, 0, 255)
            a, b = tuple(a.astype(int)), tuple(b.astype(int))
            cv2.circle(frame, a, 4, pink, -1)
            cv2.line(frame, a, b, pink, 1)
        cv2.circle(frame, b, 4, pink, -1)

        print(cam2base.dtype)
        tcp = keypoints["link_tcp"].astype(np.float64)
        cal.draw_pose_axes(
            frame,
            tcp[:3, :3],
            tcp[:3, 3],
        )

        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
        cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main(tyro.cli(Config))
