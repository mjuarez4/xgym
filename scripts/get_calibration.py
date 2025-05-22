from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import List, Optional, Union

import cv2
import numpy as np
from rich.pretty import pprint
import tyro
from xarm.wrapper import XArmAPI

np.set_printoptions(suppress=True, precision=2)


class MyCamera:
    def __repr__(self) -> str:
        return f"MyCamera(device_id={'TODO'})"

    def __init__(self, cam: Union[int, cv2.VideoCapture]):
        self.cam = cv2.VideoCapture(cam) if isinstance(cam, int) else cam
        self.thread = None

        self.fps = 30
        self.freq = 5  # hz
        self.dt = 1 / self.fps

        # while
        # _, self.img = self.cam.read()
        self.start()

    def start(self):
        self._recording = True

        def _record():
            while round(time.time(), 4) % 1:
                continue  # begin syncronized

            while self._recording:
                tick = time.time()
                # ret, self.img = self.cam.read()
                self.cam.grab()

                toc = time.time()
                elapsed = toc - tick
                time.sleep(max(0, self.dt - elapsed))

        self.thread = threading.Thread(target=_record, daemon=True)
        self.thread.start()

    def stop(self):
        self._recording = False
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            del self.thread

    def read(self):
        ret, img = self.cam.retrieve()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class OrientMode(str, Enum):
    RPY = "rpy"  # roll-pitch-yaw
    AA = "aa"  # axis-angle
    QUAT = "quat"  # quaternion
    MAT = "mat"  # rotation matrix


@dataclass
class Orient:
    mat: np.ndarray


class Xarm:
    def __init__(self, ip: str):
        self.ip = ip
        self.is_radian = True
        self.relative = False
        self.orient = OrientMode.AA

        self.robot = XArmAPI(ip, is_radian=self.is_radian)
        self.startup()

        self.ready = None
        self.position = None
        self.gripper = None

        self.speed: Speed = SPEED_JOINTS

        # if camera is not None:
        # self.camera.start()

    def startup(self):
        self.robot.connect()
        self.robot.motion_enable(enable=True)
        self.robot.set_teach_sensitivity(1, wait=True)
        self.robot.set_mode(0)  # position mode

        self.mode = 0
        self.robot.set_state(0)
        self.robot.set_gripper_enable(True)
        self.robot.set_gripper_mode(0)
        self.robot.set_gripper_speed(2000)
        self.robot.set_gripper_position(800, wait=False)

    def set_joints(self, joints: List[float]):
        assert len(joints) == 7, "Joints must be a list of 7 floats"
        ret = self.robot.set_servo_angle(
            None,
            joints,
            **self.speed.create(),
            is_radian=self.is_radian,
            relative=self.relative,
            wait=True,
        )

    @property
    def angles(self):
        return np.array(self.robot.angles)

    @property
    def pose(self):
        if self.orient == OrientMode.RPY:
            rpy = np.array(self.robot.position, dtype=np.float32)
            return rpy
        if self.orient == OrientMode.AA:
            aa = np.array(self.robot.position_aa, dtype=np.float32)
            return aa
        if self.orient == OrientMode.QUAT:
            raise NotImplementedError("Quaternion not implemented")

    @property
    def dh(self):
        """Get the Denavit-Hartenberg parameters."""
        code, dh = self.robot.get_dh_params()
        return np.array(dh, dtype=np.float32).reshape(7, 4)


@dataclass
class Speed:
    v: float = 0.5
    a: float = 0.5
    mvtime: Optional[float] = None

    def create(self):
        return {"speed": self.v, "mvacc": self.a, "mvtime": self.mvtime}


SPEED_JOINTS = Speed(
    v=np.pi / 4,  # rad/s
    a=np.pi / 3,  # rad/s^2
    mvtime=None,
)
SPEED_CARTESIAN = Speed(
    v=50,  # mm/s
    a=np.pi,  # rad/s^2
    mvtime=None,
)


@dataclass
class State:
    pass


@dataclass
class JoinState(State):
    state: List[float]

    def __post_init__(self):
        assert len(self.state) == 7, "State must be 7 floats"


@dataclass
class Config:
    ip: str = "192.168.1.231"


def deg2rad(deg: Union[float, List[float]]) -> Union[float, List[float]]:
    """Convert degrees to radians."""
    if isinstance(deg, list):
        return [d * np.pi / 180 for d in deg]
    return deg * np.pi / 180


def interpolate_waypoints(a, b, interval=0.1):
    """
    Interpolate between two waypoints a and b with a given interval.
    """
    a = np.array(a)
    b = np.array(b)

    # Calculate the number of steps needed
    steps = int(np.ceil(np.linalg.norm(b - a) / interval))

    # Generate the interpolated waypoints
    return [a + (b - a) * (i / steps) for i in range(steps)]


def main(cfg: Config):
    xarm = Xarm(cfg.ip)
    xarm.speed = SPEED_JOINTS

    print(xarm.robot.get_inverse_kinematics)
    print(xarm.robot.get_forward_kinematics)
    pprint(xarm.dh)

    joints = xarm.angles
    pprint(joints)

    ready = JoinState(state=[0.078, -0.815, -0.056, 0.570, -0.042, 1.385, 0.047])

    waypoints = [
        # ready,
        np.array([0.078, -0.815, -0.056, 0.570, -0.042, 1.385, deg2rad(183.7)]),
        np.array([-0.2, 0.17, 0.22, 1.44, -0.04, 1.27, 3.19]),
        np.array([0.54, 0.4, -0.17, 1.51, -0.81, 1.58, 3.63]),
        np.array([0.29, 0.58, -0.5, 1.49, 0.32, 1.0, 3.34]),
        np.array([-0.09, 0.29, -0.24, 1.49, 0.69, 1.49, 2.57]),
        np.array([-0.41, 0.21, -0.09, 1.25, 1.06, 1.13, 1.79]),
        np.array([-0.43, 0.44, -0.04, 1.18, 1.23, 0.98, 1.48]),
        np.array([-0.1, 0.52, -0.02, 1.48, 0.01, 0.96, 3.03]),
        np.array([0.01, 0.52, -0.15, 1.48, 0.09, 0.96, 2.38]),
        np.array([0.11, 0.89, -0.41, 1.8, 0.67, 0.59, 3.07]),
        np.array([-0.12, 0.78, -0.31, 1.67, 0.64, 0.62, 2.48]),
        np.array([-0.12, 0.78, -0.31, 1.67, 0.64, 0.62, 2.48]),
        np.array([0.77, 0.96, -0.76, 1.75, 0.49, 0.69, 3.79]),
        np.array([0.29, 0.58, -0.5, 1.49, 0.32, 1.0, 3.34]),
        np.array([-0.47, -0.1, 0.33, 0.66, -0.1, 1.12, 2.71]),
    ]

    # Interpolate waypoints
    waypoints = [
        interpolate_waypoints(a, b) for a, b in zip(waypoints[:-1], waypoints[1:])
    ]
    waypoints = sum(waypoints, [])

    frames, joints, poses = [], [], []

    cam = MyCamera(0)
    # for way in waypoints[:1]:
    for way in waypoints:
        xarm.set_joints(way)
        # cameras.snap()

        time.sleep(0.5)
        frame = cam.read()

        frames.append(frame)
        joints.append(xarm.angles)
        poses.append(xarm.pose)

        pprint(xarm.pose)

        cv2.imshow(
            "frame", cv2.resize(frame.copy(), (frame.shape[1] * 2, frame.shape[0] * 2))
        )
        cv2.waitKey(1)

    np.savez(
        "ways.npz",
        frames=frames,
        joints=joints,
        poses=poses,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
