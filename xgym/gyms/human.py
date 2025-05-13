import gymnasium as gym
import random
import time
import numpy as np
from pynput import keyboard

from xgym.controllers import KeyboardController
from xgym.gyms.base import Base


from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS

from typing import Union


class Human(Base):
    def __init__(self, mode: Union[None, "manual"] = None):
        super().__init__()

        self.mode = mode
        self._proceed = False
        self.kb = KeyboardController()
        self.kb.register(keyboard.Key.space, lambda: self.stop(toggle=True))
        self.kb.register(keyboard.Key.enter, lambda: self.proceed())

        ready = RS(cartesian=[400, -125, 350], aa=[np.pi, 0, -np.pi / 2])
        ready = self.kin_inv(np.concatenate([ready.cartesian, ready.aa]))
        self.ready = RS(joints=ready)

        self.boundary = bd.AND(
            [
                bd.CartesianBoundary(
                    min=RS(cartesian=[350, -350, -20]),
                    max=RS(cartesian=[500, -125, 300]),
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
        # logger.info("Boundaries initialized.")
        self.boundary = bd.Identity()

        # nice points for the realsense camera
        self.p1 = RS(
            cartesian=np.array([-18.071, -299.352, 565.753], dtype=np.float32),
            gripper=501,
            joints=np.array(
                [
                    -2.7980556,
                    -0.33014718,
                    0.35454515,
                    1.7675083,
                    0.82591444,
                    2.281896,
                    -1.919288,
                ],
                dtype=np.float32,
            ),
            aa=np.array([2.992124, -0.585895, -0.023022], dtype=np.float32),
        )

        self.p2 = RS(
            cartesian=np.array([586.027, -107.004, 369.824], dtype=np.float32),
            gripper=500,
            joints=np.array(
                [
                    0.05675729,
                    0.8129945,
                    -0.11665157,
                    2.720626,
                    -0.37678212,
                    2.0083778,
                    1.7185589,
                ],
                dtype=np.float32,
            ),
            aa=np.array([-3.133492, -0.442044, -1.893713], dtype=np.float32),
        )

    def reset(self):
        ret = super().reset()

        p = random.choice([self.p1, self.p2])
        p = self.p1
        self._go_joints(p, relative=False)

    def _step(self, action):
        super()._step(action)

    def wait_proceed(self):
        while not self._proceed:
            time.sleep(0.1)
        self._proceed = False

    def proceed(self):
        self._proceed = True
