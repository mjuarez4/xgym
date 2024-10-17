import logging
import time
from abc import ABC, abstractmethod
from typing import List

import cv2
import gym as oaigym
from gym import spaces

_ = None
import gymnasium as gym
import numpy as np
# from misc.boundary import BoundaryManager
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.data_utils.keyboard_interface import KBReset
from xarm.wrapper import XArmAPI

from xgym import logger
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS

# from misc.robot import XarmRobot


logger = logging.getLogger("xgym")


# define an out of bounds error
class OutOfBoundsError(Exception):
    pass


def list_cameras():
    index = 0
    arr = []
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            logger.debug(f"Camera {index} is available.")
            arr.append(index)
            cap.release()
        else:
            logger.warn(f"No camera found at index {index}.")
        index += 1
    return arr


def clear_camera_buffer(camera, nframes=5):
    for _ in range(nframes):
        camera.grab()


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

        # TODO make the observation space flexible to num_cameras

        self.imsize = 640
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )  # xyzrpyg
        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Dict(
                    {
                        "joints": spaces.Box(
                            low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32
                        ),
                        "position": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                        ),
                    }
                ),
                "img": spaces.Dict(
                    {
                        "camera_0": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.imsize, self.imsize, 3),
                            dtype=np.uint8,
                        ),
                        "wrist": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.imsize, self.imsize, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
            }
        )

        print("Initializing Base class.")
        # Initialize Robot
        logger.info("Initializing robot.")
        robot_ip = "192.168.1.231"
        self.robot = XArmAPI(robot_ip, is_radian=True)
        self.robot.connect()
        self.robot.motion_enable(enable=True)
        self.robot.set_teach_sensitivity(1, wait=True)
        self.robot.set_mode(0)
        self.robot.set_state(0)
        self.robot.set_gripper_enable(True)
        self.robot.set_gripper_mode(0)
        self.robot.set_gripper_speed(1000)
        self.robot.set_gripper_position(800, wait=False)
        logger.info("Robot initialized.")

        # Initialize Camera
        self._init_cameras()

        # Initialize Boundaries
        logger.info("Initializing boundaries.")
        self.start_angle = np.array([np.pi, 0, -np.pi / 2])
        self.boundary = bd.AND(
            [
                bd.CartesianBoundary(
                    min=RS(cartesian=[350, -350, 1]),
                    max=RS(cartesian=[500, -75, 300]),
                ),
                bd.AngularBoundary(
                    min=RS(
                        aa=np.array([-np.pi / 4, -np.pi / 4, -np.pi / 2])
                        + self.start_angle
                    ),
                    max=RS(
                        aa=np.array([np.pi / 4, np.pi / 4, np.pi / 2])
                        + self.start_angle
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
                "v": np.pi / 4,  
                "a": np.pi / 8, 
                "mvtime": None,
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
        for cam in self.cams:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.imsize)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imsize)

            cam.set(cv2.CAP_PROP_FPS, 60)

            # Disable autofocus
            # cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 to disable
            # manual focus (0 - 255, where 0 is near, 255 is far)
            # cam.set(cv2.CAP_PROP_FOCUS, 255)

            # Disable autoexposure
            # cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            # cam.set(cv2.CAP_PROP_EXPOSURE, -4)

    def _go_joints(self, pos: RS, relative=False, is_radian=True):
        logger.debug(f"Moving to position: {self.kin_fwd(pos.joints)}")
        ret = self.robot.set_servo_angle(
            None,
            pos.joints,
            speed=self.motion["joint"]["v"],
            mvacc=self.motion["joint"]["a"],
            # mvtime=self.motion["joint"]["mvtime"],
            is_radian=is_radian,
            relative=relative,
            wait=True,
            # timeout=3,
        )
        logger.debug(f"Return code: {ret}")

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

    def observation(self):
        pos = self.position
        imgs = self.look()
        self._obs = {
            "robot": {
                "joints": self.robot.angles,
                "position": pos.to_vector(),
            },
            "img": {**imgs},
        }
        return self._obs

    @abstractmethod
    def reset(self):

        self.robot.set_gripper_position(800, wait=False)
        # go to the initial position
        _, self.initial = self.robot.get_initial_point()
        self.initial = RS(joints=self.initial)
        # print(f"Initial position: {self.initial}")
        # print(self.kin_fwd(self.initial.joints))

        time.sleep(0.1)
        self._go_joints(self.ready, relative=False)
        time.sleep(0.1)
        return self.observation()

    def step(self, action):
        time.sleep(0.01)
        try:
            clipped = self.safety_check(action)
            action = clipped.to_vector() if clipped is not None else action
            self._step(action)
            return self.observation(), np.array(0.0, dtype=np.float64), False, {}
        except OutOfBoundsError as e:
            print(e)
            return self.observation(), np.array(-1.0, dtype=np.float64), True, {}

    @abstractmethod
    def _step(self, action):

        assert len(action) == 7

        act = RS.from_vector(action)
        pos = self.position
        new = pos + act

        joints = self.kin_inv(np.concatenate([new.cartesian, new.aa]))
        joints = RS(joints=joints)

        self.robot.set_gripper_position(new.gripper, wait=True)
        # return self.go( new, relative=False, is_radian=True)
        return self._go_joints(joints, relative=False)

    def look(self):
        image, depth = self.rs.read()
        size = (self.imsize, self.imsize)
        image = cv2.resize(image, size)
        out = {"wrist": image}

        for i, cam in enumerate(self.cams):
            clear_camera_buffer(cam)
            ret, img = cam.read()
            img = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
            # image = np.concatenate([image, img], axis=1)
            out[f"camera_{i}"] = img

        return out

    @abstractmethod
    def render(self, mode="human", refresh=False):
        """return the camera images from all cameras
        either cv2.imshow for 0.1 seconds or return the images as numpy arrays
        """

        if refresh:  # by default use the old camera images
            self.observation()
        imgs = np.concatenate(list(self._obs["img"].values()), axis=1)
        if mode == "human":
            cv2.imshow("Gym Environment", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)  # 1 ms delay to allow for rendering
        elif mode == "rgb_array":
            return imgs

    def safety_check(self, action):
        logger.debug(self.position)
        act = RS.from_vector(action)
        logger.warn(self.boundary.contains(self.position + act))

        if not self.boundary.contains(self.position + act):
            try:
                clipped =  self.boundary.clip(self.position + act)
                clipped = clipped - self.position
                logger.warn(f"clipping action: {action} to {clipped}")

                logger.info(self.boundary.contains(self.position + clipped))

                # input("safety: Press Enter to continue...")
                return clipped
            except Exception as e:
                logger.error(e)
                raise OutOfBoundsError("Out of bounds")

    @property
    def position(self):
        pos = np.array(self.robot.position, dtype=np.float32)
        return RS(
            cartesian=pos[:3],
            aa=pos[3:6],
            joints=np.array(self.kin_inv(pos), dtype=np.float32),
            gripper=self.gripper,
        )

    @property
    def gripper(self):
        code, pos = self.robot.get_gripper_position()
        logger.debug(f"GRIPPER: {pos} code={code}")
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
        """Inverse kinematics for the robot arm.
        Returns:
            np.array: The joint angles for the given pose.
        """
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

    def set_mode(self, mode):
        if mode == 2:
            self.robot.set_mode(2, detection_param=1)
        else:
            self.robot.set_mode(mode)
        self.robot.set_state(0)
        time.sleep(0.1)
        logger.info(f"mode: {self.robot.mode}|{self.robot.state}")

    def stop(self, toggle=False):

        print(toggle, self.robot.mode)

        if toggle and self.robot.mode == 2:
            self.robot.set_mode(0)
            self.robot.set_state(0)
        else:
            while self.robot.mode == 0 or self.robot.state == 0:
                self.set_mode(2)

            """
            try:
                tick = time.time()
                code = None
                while time.time() - tick < 5 and code != 0:
                    logger.debug("trying to enter manual mode")
                    code = self.robot.set_mode(2)
                    code = self.robot.set_state(0) or code
                    logger.info(f"code: {code}")
                    time.sleep(0.1)
                time.sleep(0.1)
                if self.robot.mode != 2:
                    raise ValueError(f"Manual mode failed: mode={self.robot.mode}")
            except:
                logger.critical( ValueError("Emergency stop activated"))
                self.robot.emergency_stop()
                raise ValueError("Emergency stop activated")
            """

        # self.robot.clean_error()
        # time.sleep(0.1)
        time.sleep(0.1)
        logger.info(f"MODE:{self.robot.mode} STATE:{self.robot.state}")
        time.sleep(0.1)

    def close(self):
        logger.debug("Cleaning up resources.")
        time.sleep(1)
        self._go_joints(self.ready, relative=False)
        self.stop()
        # self.robot.reset()
        # self.robot.disconnect()

        # self.cameras.close()
        # self.boundary.close()
        # anything else?
        pass
