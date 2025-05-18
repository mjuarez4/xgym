import logging
import os
import os.path as osp
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Union

import cv2
import gym as oaigym
from gym import spaces

from xgym.utils import camera as cu

_ = None
import gymnasium as gym
import numpy as np
from gello.cameras.camera import CameraDriver

# from misc.boundary import BoundaryManager
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.data_utils.keyboard_interface import KBReset
from xarm.wrapper import XArmAPI

from xgym import logger
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS

# from misc.robot import XarmRobot


logger = logging.getLogger("xgym")
logger.setLevel(logging.INFO)


# define an out of bounds error
class OutOfBoundsError(Exception):
    pass


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} took {end-start} seconds.")
        return result

    return wrapper


@timer
def clear_camera_buffer(camera: cv2.VideoCapture, nframes=0):
    for _ in range(nframes):
        camera.grab()


class MyCamera(CameraDriver):
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
        logger.info("start record")

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
        logger.info("stop record")
        self._recording = False
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            del self.thread

    def read(self):
        ret, img = self.cam.retrieve()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class Base(gym.Env):
    # class Base(gym.Env, ABC):
    """Base class for all LUC environments."""

    @abstractmethod
    def __init__(
        self,
        render_mode="human",
        out_dir=".",
        space="cartesian",
        # robot: XArmAPI, boundary: bd.Boundary, cameras: List[RealSenseCamera]
    ):
        super().__init__()
        self.logger = logger

        # TODO make the observation space flexible to num_cameras

        self.space = space
        self.imsize = 640
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )  # xyzrpyg

        """
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
        """

        print("Initializing Base.")
        # Initialize Robot
        logger.info("Initializing robot.")
        robot_ip = "192.168.1.231"
        self.robot = XArmAPI(robot_ip, is_radian=True)
        self.robot.connect()
        self.robot.motion_enable(enable=True)
        self.robot.set_teach_sensitivity(1, wait=True)
        self.robot.set_mode(0)
        self.mode = 0
        self.robot.set_state(0)
        self.robot.set_gripper_enable(True)
        self.robot.set_gripper_mode(0)
        self.robot.set_gripper_speed(2000)
        self.robot.set_gripper_position(800, wait=False)
        logger.info("Robot initialized.")

        self.GRIPPER_MAX = 850
        self.GRIPPER_MIN = 10

        # Initialize Camera
        self._init_cameras()

        # Initialize Boundaries
        logger.info("Initializing boundaries.")
        self.start_angle = np.array([np.pi, 0, -np.pi / 2])
        self.boundary = bd.AND(
            [
                bd.CartesianBoundary(
                    min=RS(cartesian=[150, -350, 55]),  # 5mm safety margin
                    max=RS(cartesian=[800, 350, 800]),
                ),
                # bd.AngularBoundary(
                # min=RS(
                # aa=np.array([-np.pi / 4, -np.pi / 4, -np.pi / 2])
                # + self.start_angle
                # ),
                # max=RS(
                # aa=np.array([np.pi / 4, np.pi / 4, np.pi / 2])
                # + self.start_angle
                # ),
                # ),
                # bd.GripperBoundary(min=10, max=800),
                bd.GripperBoundary(min=self.GRIPPER_MIN / self.GRIPPER_MAX, max=1),
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
                "v": 50,
                "a": np.pi,
                "mvtime": None,
            },
            "joint": {
                "v": np.pi / 5,  # rad/s
                "a": np.pi / 5,  # rad/s^2
                "mvtime": None,
            },
        }
        # self.robot.set_joint_maxacc(self.motion["joint"]["a"], is_radian=True)

        # anything else?
        self.ready = RS(
            joints=[0.078, -0.815, -0.056, 0.570, -0.042, 1.385, 0.047]
            # rpy=[np.pi, 0, -np.pi/2]
        )

        self.thread = None
        self.episode = []
        self.nepisodes = 0
        self.out_dir = osp.expanduser(out_dir)

        self._done = False
        self.limit_torque = np.array([15, 45, 10, 25, 5, 5, 5])

    def _init_cameras(self):
        logger.info("Initializing cameras.")

        # realsense cameras
        device_ids = get_device_ids()
        if len(device_ids) == 0:
            logger.error("No RealSense devices found.")
        self.rs = RealSenseCamera(flip=False, device_id=device_ids[0])
        logger.info("Cameras initialized.")

        # logitech cameras
        cams = cu.list_cameras()
        self.cams = {k: MyCamera(cam) for k, cam in cams.items()}

        # must be manually verified if changed
        self.cams = {
            "worm": self.cams[0],
            "overhead": self.cams[2],
            "side": self.cams[10],
        }

        print(self.cams)

        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.imsize)
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imsize)

        # cam.set(cv2.CAP_PROP_FPS, 60)

        # Disable autofocus
        # cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 to disable
        # manual focus (0 - 255, where 0 is near, 255 is far)
        # cam.set(cv2.CAP_PROP_FOCUS, 255)

        # Disable autoexposure
        # cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        # cam.set(cv2.CAP_PROP_EXPOSURE, -4)

    @timer
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
        """Move the robot to a given position using cartesian"""

        assert pos.cartesian is not None  # joint control not implemented

        assert pos.cartesian is not None and pos.aa is not None
        # assert not (pos.cartesian is not None and pos.joints is not None)

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

    @timer
    def observation(self):

        imgs = self.look()
        pos = self.position

        if pos.gripper is None:
            pos.gripper = self.gripper
        if pos.gripper is None:
            pos.gripper = self._obs["robot"]["position"][-1]

        self._obs = {
            "robot": {
                "joints": pos.joints,
                "position": pos.to_vector(),
            },
            "img": {**imgs},
        }
        return self._obs

    @abstractmethod
    def reset(self):

        self._done = False
        self.set_mode(0)
        self.robot.set_gripper_position(800, wait=False)
        # go to the initial position
        _, self.initial = self.robot.get_initial_point()
        self.initial = RS(joints=self.initial)
        # print(f"Initial position: {self.initial}")
        # print(self.kin_fwd(self.initial.joints))

        self._go_joints(self.ready, relative=False)
        self.robot.set_position(
            x=250,
            y=0,
            z=250,
            roll=np.pi,
            pitch=0,
            yaw=0,  # -np.pi / 2,
            relative=False,
            wait=True,
        )
        self.ready = self.position
        logger.info(f"Ready position: {self.ready}")

        time.sleep(0.1)

        # self.stop_record()
        return self.observation()

    def flush(self):
        # write the episode to disk
        # episode is list of dicts
        logger.info("Flushing episode to disk.")
        if len(self.episode):
            import jax

            episode = jax.tree.map(
                lambda x, *y: np.stack([x, *y]), self.episode[0], *self.episode[1:]
            )

            def du_flatten(d, parent_key="", sep="."):
                items = []
                for k, v in d.items():
                    new_key = parent_key + sep + k if parent_key else k
                    if isinstance(v, dict):
                        items.extend(du_flatten(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            episode = du_flatten(episode)
            out_path = osp.join(self.out_dir, f"episode{self.nepisodes}.npz")
            np.savez(out_path, **episode)
            self.episode = []
            self.nepisodes += 1

    def start_record(self):
        logger.info("start record")

        freq = 5  # hz
        dt = 1 / freq
        self._recording = True

        def _record():
            while self._recording:
                tick = time.time()
                obs = self.observation()
                self.episode.append(obs)
                toc = time.time()
                elapsed = toc - tick
                logger.debug(f"obs: {obs['robot']['position'].astype(np.float16)}")
                time.sleep(max(0, dt - elapsed))

        self.thread = threading.Thread(target=_record, daemon=True)
        self.thread.start()

    def stop_record(self):
        logger.info("stop record")
        self._recording = False
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish

    @timer
    def step(self, action):
        if self.mode == 2:
            try:
                raise ValueError("Robot in manual mode")
            except ValueError as e:
                logger.error(e)
                return (
                    self.observation(),
                    np.array(-1.0, dtype=np.float64),
                    self._done,
                    {},
                )

        try:
            # action = self.safety_check(action)
            self._step(action, wait=self.mode == 0)
            obs = self.episode[-1] if len(self.episode) else self.observation()
            return self.observation(), np.array(0.0, dtype=np.float64), self._done, {}
        except Exception as e:
            print(e)
            obs = self.episode[-1] if len(self.episode) else self.observation()
            return obs, np.array(-1.0, dtype=np.float64), self._done, {}

    @timer
    @abstractmethod
    def _step(self, action, force_grip=False, wait=True):

        if self.space == "cartesian":
            assert len(action) == 7
            act = RS.from_vector(action)
            gripper = act.gripper
        else:
            gripper = action[-1]

        if abs(gripper - self.gripper) > 0.05 or force_grip:  # save time?
            self.robot.set_gripper_position(gripper * self.GRIPPER_MAX, wait=False)
        else:
            logger.info("skipping gripper for speed")

        if self.space == "joint":
            # self._go_joints(self.position+ RS(joints=action[:-1]), relative=False, is_radian=True)
            self._go_joints(RS(joints=action[:-1]), relative=True, is_radian=True)
            return  # no cartesian move

        print(f"waiting: {wait}")
        if wait:
            self.robot.set_position(
                x=act.cartesian[0],
                y=act.cartesian[1],
                z=act.cartesian[2],
                roll=act.aa[0],
                pitch=act.aa[1],
                yaw=act.aa[2],
                relative=True,
                # speed=60,
                wait=wait,
            )
        else:  # no wait for online planning
            pos = self.position
            print(f"rpy: {pos.aa} -> {act.aa}")

            self.robot.set_position(
                x=act.cartesian[0] + self.cartesian[0],
                y=act.cartesian[1] + self.cartesian[1],
                z=act.cartesian[2] + self.cartesian[2],
                roll=act.aa[0] + pos.aa[0],
                pitch=act.aa[1] + pos.aa[1],
                yaw=act.aa[2] + pos.aa[2],
                relative=False,
                # speed=100 is default,
                mvacc=1500,  # 2000 appears to be default for cartesian
                wait=wait,
            )

        # self.safety_torque_limit()

    def send(self, action):
        """only for spacemouse"""

        # save time?
        g = action[-1]
        if g < 0.9 or (self.gripper / self.GRIPPER_MAX) < 0.9:
            self.robot.set_gripper_position(g * self.GRIPPER_MAX, wait=False)
        else:
            logger.info("skipping gripper for speed")

        self.set_mode(5)
        assert self.mode == 5
        self.robot.vc_set_cartesian_velocity(
            action[:-1],
            is_radian=True,
            is_tool_coord=False,
            duration=-1,
            # **kwargs,
        )

    @timer
    def look(self):
        size = (self.imsize, self.imsize)
        image, depth = self.rs.read()
        image = cv2.resize(image, size)
        imgs = {f"{k}": cam.read() for k, cam in self.cams.items()}
        imgs["wrist"] = image

        imgs = {k: cv2.resize(cu.square(v), size) for k, v in imgs.items()}

        # image = np.concatenate([image, img], axis=1)
        return imgs

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

    def safety_torque_limit(self):
        t = np.abs(self.torque)
        if np.any(t > self.limit_torque):
            logger.error(f"torque limit reached: {t}")
            self.set_mode(2)
            time.sleep(0.01)
            self.set_mode(2)
            raise ValueError("Torque limit exceeded")

    def safety_check(self, action):

        # self.safety_torque_limit()
        # logger.debug(self.position)

        act = RS.from_vector(action)
        new = self.position + act
        new.gripper = act.gripper

        # logger.warn(self.boundary.contains(new))
        # logger.warn(f"{self.position.aa}+{act.aa}={new.aa}")

        if not self.boundary.contains(new):
            try:
                _clipped = self.boundary.clip(new)
                clipped = _clipped - self.position
                clipped.gripper = _clipped.gripper
                logger.warn(f"clipping action: A to B")
                logger.warn(f"{action.round(4)}")  # just np
                logger.warn(f"{clipped.to_vector().round(4)}")

                # logger.info(self.boundary.contains(self.position + clipped))
                # input("safety: Press Enter to continue...")
                return clipped.to_vector()
            except Exception as e:
                logger.error(e)
                raise OutOfBoundsError("Out of bounds")

        return action

    @property
    def position(self):
        pos = np.array(self.robot.position, dtype=np.float32)
        # pos[3:6] = pos[3:6] *np.pi / 180 # deg to rad
        return RS(
            cartesian=pos[:3],
            aa=pos[3:6],
            joints=self.robot.angles,
            # np.array(self.kin_inv(pos), dtype=np.float32),
            gripper=self.gripper,
        )

    @property
    def gripper(self):
        code, pos = self.robot.get_gripper_position()
        logger.debug(f"GRIPPER: {pos} code={code}")
        pos = self.gripper if pos is None else pos
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
    def torque(self):
        return np.array(self.robot.joints_torque)

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

    @timer
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
        """Forward kinematics for the robot arm.
        Returns:
            np.array: The pose for the given joint angles.
        """
        code, pose = self.robot.get_forward_kinematics(angles)
        if code == 0:
            return pose
        else:
            raise ValueError(f"Error in forward kinematics: {code}")

    def set_mode(self, mode):
        if mode == 2:
            code = self.robot.set_mode(2, detection_param=1)
            print(f"set_mode: code={code}")
            self.robot.set_teach_sensitivity(1, wait=True)
        else:
            self.robot.set_mode(mode)

        self.robot.set_state(0)
        self.mode = mode
        time.sleep(0.01)
        logger.info(f"mode: {self.robot.mode} | state: {self.robot.state}")

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
