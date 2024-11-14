import os
import threading
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, Optional

import jax
import json_numpy
# import keyboard
import numpy as np
import pynput
# spacemouse imports
import pyspacemouse
import requests
from pynput.keyboard import Key
from pyspacemouse.pyspacemouse import SpaceNavigator


@dataclass
class SpaceMouseConfig:

    # sensitivity
    scale = np.array([
        5.0,
        5.0,
        5.0,
        0.02,
        0.02,
        0.02,
        100,
    ])
    flip = [1,1,1,-1,-1,-1,1]
    order = [0, 1, 2, 3, 5, 4, 6]

class VelocitySMC(SpaceMouseConfig):

    # xyz, rpy, gripper
    # rpy is rotation along x, y, z
    scale = np.array([
        100.0,
        100.0,
        100.0,
        0.5,
        0.5,
        0.5,
        100,
    ])
    flip = [1,1,1,-1,1,-1,1]
    order = [0, 1, 2, 3, 5, 4, 6]


def ema_2d(data, window):
    """ Calculate EMA for 2D array """

    alpha = 2 / (window + 1)  # Smoothing factor
    ema = np.empty_like(data)
    ema[0, :] = data[0, :]  # Start with first row 

    for i in range(1, data.shape[0]):
        ema[i, :] = alpha * data[i, :] + (1 - alpha) * ema[i - 1, :]

    return ema

class SpaceMouseController:

    def __init__(self, cfg: SpaceMouseConfig = VelocitySMC()):

        self.cfg = cfg
        self.state: Optional[SpaceNavigator] = SpaceNavigator(
            t=0.0,
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            buttons=(0, 0),
        )

        self._running = False
        self.thread = None

        self.freq = 300  # Frequency Hz
        self.dt = 1.0 / self.freq

        self.hist = np.zeros((self.freq, 7))

        self.success = pyspacemouse.open(
            dof_callback=self.set_state,
            button_callback=None,
            button_callback_arr=[],
        )

        if self.success:
            print("SpaceMouse connected.")
        else:
            print("Failed to connect to SpaceMouse.")

        self.start()

    def start(self):
        """Start reading the SpaceMouse in the background."""
        if not self._running and self.success:
            self._running = True
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        time.sleep(0.1)

    def _read(self):
        """Read input in a loop and update the last state."""
        while self._running and self.success:
            pyspacemouse.read()
            time.sleep(self.dt)

    def stop(self):
        """Stop reading and close the SpaceMouse."""
        if self._running:
            self._running = False
            if self.thread is not None:
                self.thread.join()  # Wait for the thread to finish
                del self.thread
            pyspacemouse.close()  # Close the SpaceMouse device when stopping
            print("SpaceMouse disconnected.")

    def set_state(self, state):
        self.state = state

    def read(self, as_dict=False):

        if as_dict:
            print(type(self.state))
            return self.state._asdict()

        else:  # x,y,z,roll,pitch,yaw,gripper
            gripper = self.state.buttons
            gripper = (-1 * gripper[0] + gripper[1])
            out = np.array([
                self.state.x,
                self.state.y,
                self.state.z,
                self.state.pitch,
                self.state.yaw,
                self.state.roll,
                gripper,
            ])
            out = out[self.cfg.order] 
            out = out ** 2 * np.sign(out)
            out = out*self.cfg.scale*self.cfg.flip

            self.hist = np.roll(self.hist, -1, axis=0) # smooth
            self.hist[-1] = out
            # out = ema_2d(self.hist, 10)[-1]

            # out = np.where(out > self.hist[-2], out, 0)
            return out


class Controller(ABC):

    def __init__(self):
        pass


class KeyboardController:
    def __init__(self):

        self.vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.running = True
        self.keys_pressed = set()
        self.listener = pynput.keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )

        self.size = 10.0
        self.rsize = 0.05
        self.gsize = 50.0

        self.funcs = {}

        self.run()  # Begin the run method at the end of initialization

    def on_key_press(self, key):

        if isinstance(key, pynput.keyboard._xorg.KeyCode):
            key = key.char

        if key in self.funcs:
            self.funcs[key]()
        self.keys_pressed.add(key)
        self.update_vector()

    def on_key_release(self, key):

        if isinstance(key, pynput.keyboard._xorg.KeyCode):
            key = key.char
        self.keys_pressed.discard(key)
        self.update_vector()

    def update_vector(self):
        self.vec = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # Reset vec to 0 at the beginning

        shift_pressed = (
            Key.shift in self.keys_pressed or Key.shift_r in self.keys_pressed
        )

        if Key.up in self.keys_pressed:
            if shift_pressed:
                self.vec[4] += self.rsize  # Increase ry value
            else:
                self.vec[1] += self.size  # Increase y value
        if Key.down in self.keys_pressed:
            if shift_pressed:
                self.vec[4] -= self.rsize  # Decrease ry value
            else:
                self.vec[1] -= self.size  # Decrease y value
        if Key.left in self.keys_pressed:
            if shift_pressed:
                self.vec[3] -= self.rsize  # Decrease rx value
            else:
                self.vec[0] -= self.size  # Decrease x value
        if Key.right in self.keys_pressed:
            if shift_pressed:
                self.vec[3] += self.rsize  # Increase rx value
            else:
                self.vec[0] += self.size  # Increase x value

        if any(hasattr(k, "char") and k.char == "w" for k in self.keys_pressed):
            self.vec[2] += self.size  # Increase z value
        if any(hasattr(k, "char") and k.char == "s" for k in self.keys_pressed):
            self.vec[2] -= self.size  # Decrease z value
        if any(hasattr(k, "char") and k.char == "e" for k in self.keys_pressed):
            self.vec[5] += self.rsize  # Increase rz value
        if any(hasattr(k, "char") and k.char == "d" for k in self.keys_pressed):
            self.vec[5] -= self.rsize  # Decrease rz value

        if any(hasattr(k, "char") and k.char == "m" for k in self.keys_pressed):
            self.vec[6] += self.gsize  # Increase gripper value (open)
        if any(hasattr(k, "char") and k.char == "n" for k in self.keys_pressed):
            self.vec[6] -= self.gsize  # Decrease gripper value (close)

        self.vec = np.array(self.vec)

    def run(self):
        print("running")
        thread = threading.Thread(target=self.listener.start, daemon=True)
        thread.start()
        """
        try:
            while True:
                # print(self.vec)  # Continuously print the vector to see the changes
                time.sleep(0.1)  # Print every 0.1 second
        except KeyboardInterrupt:
            self.running = False
            self.listener.stop()
        """

    def close(self):
        self.running = False
        self.listener.stop()

    def __call__(self, *args, **kwargs):
        return self.vec

    def register(self, key, func):
        self.funcs[key] = func


# kb = KeyboardController()
# while True:
# time.sleep(0.1)


class ScriptedController(Controller):
    def __init__(self):
        self.action = None

    def __call__(self, obs):
        return self.action

    def update(self, obs, reward, truncated, terminated, info):
        self.action = 1


class ModelController(Controller):
    def __init__(self, ip, port, ensemble=True):
        self.server = ip
        self.port = port
        self.url_query = f"http://{self.server}:{self.port}/query"
        self.url_reset = f"http://{self.server}:{self.port}/reset"
        self.ensemble = ensemble

    def __call__(self, primary, high=None, wrist=None):

        payload = {
            "observation": {
                "image_primary": primary.tolist(),
                **({"wrist": wrist.tolist()} if wrist is not None else {}),
                **({"high": high.tolist()} if high is not None else {}),
            },
            "modality": "l",  # can we use both? there is another letter for both
            "ensemble": self.ensemble,
            "model": "bafl",
            "dataset_name": "xgym_single",  # Ensure this matches the server's dataset_name
        }

        spec = lambda x: jax.tree.map(lambda arr: type(arr), x)
        # pprint(spec(payload))
        # quit()

        out = self.send_observation_and_get_action(payload)
        return out

    def send_observation_and_get_action(self, payload):
        """note from klaud
        they json-ify the data twice
        """

        # Set timeout to avoid hanging
        response = requests.post(self.url_query, json=payload, timeout=10)
        response.raise_for_status()
        response_text = response.text
        action = json_numpy.loads(response_text)  # 2x json-ified
        action = json_numpy.loads(action)
        return action

    def reset(self):
        payload = {
            # "observation": {"image_primary": image.tolist()},
            "text": "put the yellow block on the green block",
            "modality": "l",  # can we use both? there is another letter for both
            "ensemble": self.ensemble,
            "model": "bafl",
            "dataset_name": "xgym_single",  # Ensure this matches the server's dataset_name
        }

        response = requests.post(self.url_reset, json=payload, timeout=10)
        response.raise_for_status()
        return response.text


def build_controller(mode="scripted"):
    if mode == "scripted":
        return ScriptedController()
    elif mode == "model":
        return ModelController()
    else:
        raise ValueError(f"Invalid mode: {mode}")


def main():
    sm = SpaceMouseController()
    while True:
        print(sm.read().round(4))
        time.sleep(0.1)


if __name__ == "__main__":
    main()
