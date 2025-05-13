import json_numpy as jp

jp.patch()
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, Optional

import jax

# import keyboard
import numpy as np

# spacemouse imports
import requests


class ModelController:
    def __init__(self, ip, port, task, ensemble=True):
        self.server = ip
        self.port = port
        self.url_query = f"http://{self.server}:{self.port}/query"
        self.url_reset = f"http://{self.server}:{self.port}/reset"
        self.ensemble = ensemble

        self.tasks = {
            "duck": {
                "text": "put the ducks in the bowl",
                "dataset_name": "xgym_duck_single",
            },
            "stack": {
                "text": "stack all the blocks vertically ",
                "dataset_name": "xgym_stack_single",
            },
            "lift": {
                "text": "pick up the red block",
                "dataset_name": "xgym_lift_single",
            },
            "play": {"text": "pick up any object", "dataset_name": "xgym_play_single"},
        }
        self.task = task

    def __call__(self, primary, high=None, side=None, wrist=None, proprio=None):

        image = {
            f"image_{k}": v.tolist()
            for k, v in {
                "primary": primary,
                "left_wrist": wrist,
                "high": high,
                "side": side,
            }.items()
            if v is not None
        }

        dataset_name = self.tasks[self.task]["dataset_name"]
        payload = {
            "observation": {
                **image,
                **({"proprio_single": proprio} if proprio is not None else {}),
            },
            "modality": "l",  # can we use both? there is another letter for both
            "ensemble": self.ensemble,
            "model": "bafl",
            "dataset_name": dataset_name,  # Ensure this matches the server's dataset_name
        }

        spec = lambda x: jax.tree.map(lambda arr: type(arr), x)
        # pprint(spec(payload))
        # quit()

        payload = jax.tree.map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x, payload
        )
        out = self.send_observation_and_get_action(payload)
        out = np.array([out]) if self.ensemble else out
        return out

    def send_observation_and_get_action(self, payload):
        """note from klaud
        they json-ify the data twice
        """

        # Set timeout to avoid hanging
        response = requests.post(self.url_query, json=payload, timeout=15)
        response.raise_for_status()
        response_text = response.text
        action = jp.loads(response_text)  # 2x json-ified
        action = jp.loads(action)
        return action

    def reset(self):
        dataset_name = self.tasks[self.task]["dataset_name"]
        payload = {
            # "observation": {"image_primary": image.tolist()},
            "text": self.tasks[self.task]["text"],
            "modality": "l",  # can we use both? there is another letter for both
            "ensemble": self.ensemble,
            "model": "bafl",
            "dataset_name": dataset_name,  # Ensure this matches
        }

        response = requests.post(self.url_reset, json=payload, timeout=15)
        response.raise_for_status()
        return response.text


import traceback

import xgym


class HamerController:
    def __init__(self, ip, port):
        self.server = ip
        self.host = ip
        self.port = port
        self.url_query = f"http://{self.server}:{self.port}/query"
        self.url_reset = f"http://{self.server}:{self.port}/reset"

    def __call__(self, observation: np.ndarray):

        payload = {"observation": observation}

        # spec = lambda x: jax.tree.map(lambda arr: type(arr), x)
        # pprint(spec(payload))
        # payload = jp.dumps(payload)

        for r in range(3):
            out = self.send_observation_and_get_action(payload)
            if out != "error" and out != {}:
                break

        if out == "error" or out == {}:
            xgym.logger.warn(f"Request failed: {self.host}:{self.port}")

        return out

    def send_observation_and_get_action(self, payload):
        """note from klaud
        they json-ify the data twice
        """

        try:
            # Set timeout to avoid hanging
            response = requests.post(self.url_query, json=payload, timeout=15)
            response.raise_for_status()

            out = jax.tree.map(jp.loads, response.text)
            if out == "error":
                return out
            out = jax.tree.map(jp.loads, out)

            if not isinstance(out, dict):
                raise ValueError(f"We have a problem")
            return out
        except Exception as e:
            xgym.logger.warn(f"Request failed: {self.host}:{self.port}")
            traceback.print_exc()
            # print(f"Request failed: {e}")
            return {}

    def reset(self):
        payload = {}
        response = requests.post(self.url_reset, json=payload, timeout=15)
        response.raise_for_status()
        return response.text
