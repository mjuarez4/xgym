"""Visualize the robot pose in the camera frame."""

from dataclasses import dataclass

import cv2
from get_calibration import SPEED_JOINTS, Xarm
from get_calibration import Config as BaseConfig
import jax
import numpy as np
import tyro

from xgym import BASE
from xgym.calibrate.april import Calibrator
from xgym.calibrate.urdf.robot import RobotTree


@dataclass
class Config(BaseConfig):
    """Configuration for :class:`ViewCalibrator`."""

    camera_id: int = 0


class ViewCalibrator:
    """Utility to draw robot joints in the camera frame."""

    def __init__(self, cfg: Config) -> None:
        self.xarm = Xarm(cfg.ip)
        self.xarm.speed = SPEED_JOINTS

        self.robot = RobotTree()
        self.calibrator = Calibrator()

        self.cap = cv2.VideoCapture(cfg.camera_id)
        self.cam2base = np.load(BASE / "base.npz")["base"]

    def _compute_keypoints(self, joints: np.ndarray) -> dict[str, np.ndarray]:
        """Return transform matrices for each joint in camera space."""

        kins = self.robot.set_pose(joints)
        mats = {k: v.get_matrix().numpy()[0] for k, v in kins.items()}
        return {k: self.cam2base @ v for k, v in mats.items()}

    def _draw_robot(self, frame: np.ndarray, keypoints: dict[str, np.ndarray]) -> None:
        """Draw the robot joints as points connected by lines."""

        def _project(kp: np.ndarray) -> np.ndarray:
            """Project a 4x4 point from camera space to image space."""
            kp = self.calibrator.project_points(
                (0, 0, 0),
                kp[:3, :3],
                kp[:3, 3],
            )
            return kp.reshape(-1)

        points = jax.tree.map(_project, keypoints)

        last = None
        for i, (k, v) in enumerate(points.items()):
            pink = (255, 0, 255)
            v = tuple(v.astype(int))
            cv2.circle(frame, v, 4, pink, -1)
            # cv2.putText(frame, k, v, cv2.FONT_HERSHEY_SIMPLEX, 0.5, pink, 1)
            # if last is not None:
            # cv2.line(frame, last, v, pink, 1)
            last = v

    def _draw_axes(self, frame: np.ndarray, keypoints: dict[str, np.ndarray]) -> None:
        """Overlay coordinate axes for the camera and TCP."""

        self.calibrator.draw_pose_axes(
            frame,
            self.cam2base[:3, :3],
            self.cam2base[:3, 3],
        )

        tcp = keypoints["link_tcp"].astype(np.float64)
        self.calibrator.draw_pose_axes(frame, tcp[:3, :3], tcp[:3, 3])

    def step(self) -> np.ndarray | None:
        """Grab one frame and annotate it."""

        ret, frame = self.cap.read()
        if not ret:
            return None

        joints = self.xarm.angles
        keypoints = self._compute_keypoints(joints)

        self._draw_axes(frame, keypoints)
        self._draw_robot(frame, keypoints)

        return frame

    def run(self) -> None:
        """Loop until the user presses ``q``."""

        while True:
            frame = self.step()
            if frame is None:
                continue

            frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
            cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) == ord("q"):
                break


def main(cfg: Config) -> None:
    ViewCalibrator(cfg).run()


if __name__ == "__main__":
    main(tyro.cli(Config))
