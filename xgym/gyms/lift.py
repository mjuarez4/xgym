import time
from typing import Union

import gymnasium as gym
import numpy as np
from pynput import keyboard

from xgym import logger
from xgym.controllers import KeyboardController
from xgym.gyms.base import Base
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS


class Lift(Base):
    def __init__(self, manual=False, out_dir=".", random=True):
        super().__init__(out_dir=out_dir)

        assert manual in [True, False]
        self.manual = manual
        self.random = random
        self._proceed = False

        self.kb = KeyboardController()
        self.kb.register(keyboard.Key.space, lambda: self.stop(toggle=True))
        self.kb.register(keyboard.Key.enter, lambda: self.proceed())

        def _set_done():
            self._done = True
            self.logger.warning("done")

        self.kb.register("r", lambda: _set_done())

        # ready = RS(cartesian=[400, -125, 350], aa=[np.pi, 0, -np.pi / 2])
        # ready = self.kin_inv(np.concatenate([ready.cartesian, ready.aa]))
        # self.ready = RS(joints=ready)

        self.boundary = bd.AND(
            [
                bd.CartesianBoundary(
                    min=RS(cartesian=[120, -450, -25]),  # -500 give kinematic error
                    max=RS(cartesian=[500, -180, 300]),  # y was -250
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
                bd.GripperBoundary(min=10 / 850, max=1),
            ]
        )

    def reset(self):
        ret = super().reset()

        if self.manual:

            print("please reset the environment")
            time.sleep(1)
            self.set_mode(2)

            print("bring the end effector to the cube")
            print("press enter to proceed")
            time.sleep(0.1)
            self.wait_proceed()
            logger.warning("proceeding")
            time.sleep(0.1)
            self.p1 = self.position

            time.sleep(0.1)
            self.set_mode(0)
            logger.warning("manual off")
            time.sleep(0.1)
            logger.warning("resetting")
            self._step(np.array([0, 0, 100, 0, 0, 0, 1]))  # go up
            ret = super().reset()
            logger.warning("resetted")
            time.sleep(0.1)

        if self.random:

            # random starting position
            mod = np.random.choice([-1, 1], size=2)
            rand = np.random.randint(10, 50, size=2) * mod
            rand[1] *= np.random.choice([1, 1.5, 2], size=1) if rand[1] < 0 else 1
            rand = rand.tolist()
            randy = -abs(np.random.randint(10, 350, size=1) * mod)

            print(rand)
            step = np.array([rand[0], randy[0], rand[1], 0, 0, 0, 1])

            self._step(step)

        return self.observation()

    def wait_proceed(self):
        while not self._proceed:
            time.sleep(0.1)
        self._proceed = False

    def proceed(self):
        self._proceed = True

    def auto_reset(self):
        """lets you collect data with no human intervention
        randomizes the position of the cube
        """
        self.set_mode(0)
        self.manual = False

        mod = np.random.choice([-1, 1], size=2)
        rand = np.random.randint(1, 25, size=2) * mod

        # random small horizontal move
        step = np.array([rand[0], rand[1], 0, 0, 0, 0, self.gripper / self.GRIPPER_MAX])
        step = self.safety_check(step)
        self._step(step)
        # place the block on the table
        # step vs _step so its safe
        self.step(np.array([0, 0, -100, 0, 0, 0, self.gripper / self.GRIPPER_MAX]))
        # release the block
        self._step(np.array([0, 0, 0, 0, 0, 0, 1]), force_grip=True)
        self.p1 = self.position

        # go up so you dont hit the block before reset
        self._step(np.array([0, 0, 100, 0, 0, 0, 1]), force_grip=True)
