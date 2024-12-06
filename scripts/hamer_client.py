import json
from json_numpy import default, object_hook
import json_numpy as jp

jp.patch()

from pprint import pprint
from typing import Any, Dict, Optional

import jax
import numpy as np
import requests
# import uvicorn
# from fastapi import FastAPI
from fastapi.responses import JSONResponse

spec = lambda x: jax.tree.map(lambda arr: type(arr), x)


def jsonify(obj):
    return JSONResponse(jp.dumps(obj))


import time 

class HamerController:
    def __init__(self, ip, port):
        self.server = ip
        self.port = port
        self.url_query = f"http://{self.server}:{self.port}/query"
        self.url_reset = f"http://{self.server}:{self.port}/reset"

    def __call__(self, observation: np.ndarray):


        payload = {"observation": observation}


        # spec = lambda x: jax.tree.map(lambda arr: type(arr), x)
        # pprint(spec(payload))
        # payload = jp.dumps(payload)

        out = self.send_observation_and_get_action(payload)
        return out

    def send_observation_and_get_action(self, payload):
        """note from klaud
        they json-ify the data twice
        """

        try:
            # Set timeout to avoid hanging
            response = requests.post(self.url_query, json=payload, timeout=15)
            response.raise_for_status()

            out = jp.loads(response.text)

            if not isinstance(out , dict):
                raise ValueError(f"We have a problem")
            return out
        except Exception as e:
            print(f"Request failed: {e}")
            return {}

    def reset(self):
        payload = {}
        response = requests.post(self.url_reset, json=payload, timeout=15)
        response.raise_for_status()
        return response.text


model = HamerController("aisec-102.cs.luc.edu", 8001)

import cv2

cap = cv2.VideoCapture(0)
# model.reset()

i = 0
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    i += 1
    # if i % 6 != 0:
    # continue

    frame = cv2.resize(frame, (224, 224))

    print(frame.shape)
    out = model(frame)

    print(type(out))
    print(out.keys())

    time.sleep(0.1)
