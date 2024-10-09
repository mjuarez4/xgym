import os
import numpy as np
import threading
import time
from abc import ABC, abstractmethod

import keyboard
from pynput import keyboard


class Controller(ABC):

    def __init__(self):
        pass


class KeyboardController:
    def __init__(self):

        self.vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.running = True
        self.keys_pressed = set()
        self.listener = keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )

        self.size = 10.0
        self.rsize = 0.05
        self.gsize = 50.0

        self.funcs = {}

        self.run()  # Begin the run method at the end of initialization

    def on_key_press(self, key):
        if key in self.funcs:
            self.funcs[key]()

        self.keys_pressed.add(key)
        self.update_vector()

    def on_key_release(self, key):
        self.keys_pressed.discard(key)
        self.update_vector()

    def update_vector(self):
        self.vec = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Reset vec to 0 at the beginning

        shift_pressed = (
            keyboard.Key.shift in self.keys_pressed
            or keyboard.Key.shift_r in self.keys_pressed
        )

        if keyboard.Key.up in self.keys_pressed:
            if shift_pressed:
                self.vec[4] += self.rsize  # Increase ry value
            else:
                self.vec[1] += self.size  # Increase y value
        if keyboard.Key.down in self.keys_pressed:
            if shift_pressed:
                self.vec[4] -= self.rsize  # Decrease ry value
            else:
                self.vec[1] -= self.size  # Decrease y value
        if keyboard.Key.left in self.keys_pressed:
            if shift_pressed:
                self.vec[3] -= self.rsize  # Decrease rx value
            else:
                self.vec[0] -= self.size  # Decrease x value
        if keyboard.Key.right in self.keys_pressed:
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


class ScriptedController(Controller):
    def __init__(self):
        self.action = None

    def __call__(self, obs):
        return self.action

    def update(self, obs, reward, truncated, terminated, info):
        self.action = 1


class ModelController(Controller):
    def __init__(self):
        self.model = None

    def __call__(self, obs):
        return self.model.predict(obs)


def build_controller(mode="scripted"):
    if mode == "scripted":
        return ScriptedController()
    elif mode == "model":
        return ModelController()
    else:
        raise ValueError(f"Invalid mode: {mode}")
