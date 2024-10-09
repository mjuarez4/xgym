import logging
import time
from abc import ABC, abstractmethod
from typing import List

import cv2
import gymnasium as gym
import numpy as np
# from misc.boundary import BoundaryManager
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.data_utils.keyboard_interface import KBReset
from xarm.wrapper import XArmAPI

from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS

# from misc.robot import XarmRobot

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# define an out of bounds error
class OutOfBoundsError(Exception):
    pass


def list_cameras():
    index = 0
    arr = []
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f"Camera {index} is available.")
            arr.append(index)
            cap.release()
        else:
            print(f"No camera found at index {index}.")
        index += 1
    return arr


class Base(gym.Env):
    # class Base(gym.Env, ABC):
    """Base class for all LUC environments."""

    @abstractmethod
    def __init__(
        self,
        render_mode="human",
        # robot: XArmAPI, boundary: bd.Boundary, cameras: List[RealSenseCamera]
    ):
        super().__init__()

        print("Initializing Base class.")
        # Initialize Robot
        logger.info("Initializing robot.")
        robot_ip = "192.168.1.231"
        self.robot = XArmAPI(robot_ip, is_radian=True)
        self.robot.connect()
        self.robot.motion_enable(enable=True)
        self.robot.set_mode(0)
        self.robot.set_state(0)
        self.robot.set_gripper_enable(True)
        self.robot.set_gripper_mode(0)
        self.robot.set_gripper_speed(1000)
        self.robot.set_gripper_position(500, wait=False)
        logger.info("Robot initialized.")

        # Initialize Camera
        self._init_cameras()

        # Initialize Boundaries
        logger.info("Initializing boundaries.")
        start_angle = np.array([np.pi, 0, -np.pi / 2])
        self.boundary = bd.AND(
            [
                bd.CartesianBoundary(
                    min=RS(cartesian=[350, -350, 5]),
                    max=RS(cartesian=[500, -75, 300]),
                ),
                bd.AngularBoundary(
                    min=RS(
                        aa=np.array([-np.pi / 4, -np.pi / 4, -np.pi / 2]) + start_angle
                    ),
                    max=RS(
                        aa=np.array([np.pi / 4, np.pi / 4, np.pi / 2]) + start_angle
                    ),
                ),
                bd.GripperBoundary(min=10, max=800),
            ]
        )
        logger.info("Boundaries initialized.")

        # self.robot.set_reduced_joint_range()
        # self.robot.set_reduced_max_joint_speed()
        # self.robot.set_servo_angle_j(0, 0, wait=True)

        # set the task description
        self.robot.set_collision_rebound(True)  # rebound on collision

        self.motion = {
            "cartesian": {
                "v": 100,
                "a": 15,
                "mvtime": 1,
            },
            "joint": {
                "v": np.pi / 4,  # rad/s
                "a": np.pi / 8,  # rad/s^2
                "mvtime": None,
            },
        }

        # anything else?
        self.ready = RS(
            joints=[
                -0.223213,
                -0.228721,
                0.021959,
                1.018191,
                0,
                1.246863,
                1.367298,
            ]
            # rpy=[np.pi, 0, -np.pi/2]
        )

    def _init_cameras(self):
        logger.info("Initializing cameras.")

        # realsense cameras
        device_ids = get_device_ids()
        if len(device_ids) == 0:
            logger.error("No RealSense devices found.")
        self.rs = RealSenseCamera(flip=False, device_id=device_ids[0])
        logger.info("Cameras initialized.")

        # logitech cameras
        idxs = list_cameras()
        self.cams = [cv2.VideoCapture(i) for i in idxs]

    def _go_joints(self, pos: RS, relative=False, is_radian=True):
        logger.info(f"Moving to position: {pos}")
        ret = self.robot.set_servo_angle(
            None,
            pos.joints,
            speed=self.motion["joint"]["v"],
            mvacc=self.motion["joint"]["a"],
            mvtime=self.motion["joint"]["mvtime"],
            is_radian=is_radian,
            relative=relative,
            wait=True,
            # timeout=3,
        )
        logger.warning(f"Return code: {ret}")

    def go(self, pos: RS, relative=True, is_radian=True, tool=False):

        assert pos.cartesian is not None  # joint control not implemented

        assert (
            pos.cartesian is not None and pos.aa is not None
        ) or pos.joints is not None
        assert not (pos.cartesian is not None and pos.joints is not None)

        x, y, z = pos.cartesian
        roll, pitch, yaw = pos.aa

        gripper = pos.gripper

        move = self.robot.set_tool_position if tool else self.robot.set_position
        code = move(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            radius=None,
            speed=self.motion["cartesian"]["v"],
            mvacc=self.motion["cartesian"]["a"],
            mvtime=self.motion["cartesian"]["mvtime"],
            relative=relative,  # relativeto pos
            is_radian=is_radian,
            wait=True,
            timeout=10,
        )

    @abstractmethod
    def reset(self):

        # go to the initial position
        _, self.initial = self.robot.get_initial_point()
        self.initial = RS(joints=self.initial)
        print(f"Initial position: {self.initial}")
        # print(self.kin_fwd(self.initial.joints))

        self._go_joints(self.ready, relative=False)

    def step(self, action):
        if np.array(action).sum() == 0:
            return
        try:
            self.safety_check(action)
            return self._step(action)
            return "ready"
        except OutOfBoundsError as e:
            print(e)
            return self.position, -1, True, False, {}

    @abstractmethod
    def _step(self, action):
        act = RS.from_vector(action)
        pos = self.position
        new = pos + act
        joints = self.kin_inv(np.concat([new.cartesian, new.aa]))
        joints = RS(joints=joints)
        print(joints)

        self.robot.set_gripper_position(new.gripper, wait=False)
        return self._go_joints(joints, relative=False)
        return self.go(
            act,
            relative=True,
            is_radian=False,
        )

    def look(self, size=(224, 224)):
        image, depth = self.rs.read()
        image = cv2.resize(image, size)

        for i, cam in enumerate(self.cams):
            ret, img = cam.read()
            img = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
            image = np.concatenate([image, img], axis=1)

        return image

    @abstractmethod
    def render(self, mode="human"):
        """return the camera images from all cameras
        either cv2.imshow for 0.1 seconds or return the images as numpy arrays
        """

        if mode == "human":
            imgs = self.look((640, 640))
            cv2.imshow("Gym Environment", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)  # 1 ms delay to allow for rendering

        elif mode == "rgb_array":
            return self.look()

    def safety_check(self, action):
        print(self.position)
        act = RS.from_vector(action)
        if not self.boundary.contains(self.position + act):
            raise OutOfBoundsError("Out of bounds")

    @property
    def position(self):
        pos = self.robot.position
        return RS(
            cartesian=pos[:3],
            aa=pos[3:],
            joints=self.kin_inv(pos),
            gripper=self.gripper,
        )

    @property
    def gripper(self):
        code, pos = self.robot.get_gripper_position()
        logger.info(f'GRIPPER: {code} {pos}')
        return pos

    @property
    def cartesian(self):
        return self.robot.position[:3]

    @property
    def aa(self):
        return self.robot.position_aa[3:]

    @property
    def rpy(self):
        return self.robot.position[3:]

    @property
    def angles(self):
        return self.robot.angles

    @property
    def state(self):
        return {
            "cartesian": {
                "cartesian": self.robot.get_position_aa()[1][:3],
                "aa": self.robot.get_position_aa()[1][3:] / 1000,  # m to mm
            },
            "angles": self.robot.angles,
            "joint": {
                "position": [self.robot.get_joint_states(num=i)[1] for i in range(7)],
                "velocity": self.robot.realtime_joint_speeds,  # deg/s or rad/s
                "torque": self.robot.joints_torque,
            },
            "speed": self.robot.last_used_joint_speed,
            "acceleration": self.robot.last_used_joint_acc,
            "temperature": self.robot.temperatures,
            "voltage": self.robot.voltages,
            "current": self.robot.currents,
            "is_moving": self.robot.is_moving,
            "gripper": {
                "position": self.robot.get_gripper_position,
            },
        }

    def set_position(self, position, wait=True, action_space="aa"):
        if action_space == "aa":
            self.robot.set_position_aa(position, wait)
        elif action_space == "joint":
            self.robot.set_joint_position(position, wait)
        else:
            raise ValueError(f"Invalid action space: {action_space}")


    def clean(self):
        # only use if specific error @ klaud figure out what these should be
        """
        Clean the error, need to be manually enabled
        motion(arm.motion_enable(True)) and set state(arm.set_state(state=0))
        after clean error

        :return: code
            code: See the [API Code Documentation](./xarm_api_code.md#api-code)
            for details.
        """
        self.robot.clean_error()
        self.robot.clean_gripper_error()
        self.robot.clean_warn()
        # @klaud do some reading to understand if robot.clean_error() is preferred
        # over robot.reset(params)

        # self.cameras.clean()
        # self.boundary.clean()
        # anything else?

    def kin_inv(self, pose):
        code, angles = self.robot.get_inverse_kinematics(pose)
        if code == 0:
            return angles
        else:
            raise ValueError(f"Error in inverse kinematics: {code}")

    def kin_fwd(self, angles):
        code, pose = self.robot.get_forward_kinematics(angles)
        if code == 0:
            return pose
        else:
            raise ValueError(f"Error in inverse kinematics: {code}")

    def stop(self, toggle=False):

        if toggle and self.robot.mode == 2:
            self.robot.set_mode(0)
            self.robot.set_state(0)
        else:
            try:
                print("trying to enter manual mode")
                tick = time.time()
                while time.time() - tick < 5 and self.robot.mode != 2:
                    self.robot.set_mode(2)
                    self.robot.set_state(0)
                if self.robot.mode != 2:
                    raise ValueError("Manual mode failed")
            except:
                self.robot.emergency_stop()
                raise ValueError("Emergency stop activated")

    def close(self):
        logger.info("Cleaning up resources.")
        time.sleep(1)
        self.robot.reset()
        self.robot.disconnect()
        # self.cameras.close()
        # self.boundary.close()
        # anything else?
        pass
