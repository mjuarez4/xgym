import gymnasium as gym
import time
import numpy as np
from pynput import keyboard

from xgym.controllers import KeyboardController
from xgym.gyms.base import Base


from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS

from typing import Union

class Stack(Base):
    def __init__(self, mode: Union[None, 'manual'] = None):
        super().__init__()

        self.mode = mode
        self._proceed = False
        self.kb = KeyboardController()
        self.kb.register(keyboard.Key.space, lambda: self.stop(toggle=True))
        self.kb.register(keyboard.Key.enter, lambda: self.proceed())

        ready = RS(cartesian=[400,-125,350], aa=[np.pi,0,-np.pi/2])
        ready = self.kin_inv(np.concatenate([ready.cartesian,ready.aa]))
        self.ready = RS(joints=ready)

        self.boundary = bd.AND(
            [
                bd.CartesianBoundary(
                    min=RS(cartesian=[350, -350, -20]),
                    max=RS(cartesian=[500, -125, 300]),
                ),
                bd.AngularBoundary(
                    min=RS(
                        aa=np.array([-np.pi / 4, -np.pi / 4, -np.pi / 2]) + self.start_angle
                    ),
                    max=RS(
                        aa=np.array([np.pi / 4, np.pi / 4, np.pi / 2]) + self.start_angle
                    ),
                ),
                bd.GripperBoundary(min=10, max=800),
            ]
        )
        # logger.info("Boundaries initialized.")
        self.boundary = bd.Identity()


    def reset(self):
        ret = super().reset()
        if self.mode != 'manual':
            return ret
        
        print("please reset the environment")
        self.stop(toggle=True)

        print("bring the end effector to the first cube")
        print("press enter to proceed")
        self.wait_proceed()
        self.p1 = self.position

        self.stop(toggle=True)
        ret = super().reset()
        self.stop(toggle=True)

        print("bring the end effector to the second cube")
        print("press enter to proceed")
        self.wait_proceed()
        self.p2 = self.position

        self.stop(toggle=True)
        return super().reset()

    def _step(self, action):
        super()._step(action)

    def render(self, mode="human"):
        super().render(mode)

    def wait_proceed(self):
        while not self._proceed:
            time.sleep(0.1)
        self._proceed = False

    def proceed(self):
        self._proceed = True
