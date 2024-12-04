import json_numpy

json_numpy.patch()
import time
import traceback
from collections import deque
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))


def resize(img, size=(224, 224)):
    img = tf.image.resize(img, size=size, method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


from pprint import pprint

from draccus.argparsing import ArgumentParser as AP
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import (DEFAULT_CHECKPOINT, HAMER, MANO, download_models,
                          load_hamer)


@dataclass
class DemoCN:
    """config node for demo"""

    checkpoint: str = DEFAULT_CHECKPOINT  # Path to pretrained model checkpoint
    img_folder: str = "example_data"  # Folder with input images
    out_folder: str = "out_demo"  # Output folder to save rendered results
    side_view: bool = False  # If set, render side view also
    full_frame: bool = True  # If set, render all people together also
    save_mesh: bool = True  # If set, save meshes to disk also
    batch_size: int = 1  # Batch size for inference/fitting
    rescale_factor: float = 2.0  # Factor for padding the bbox
    body_detector: str = "vitdet"  # Using regnety improves runtime and reduces memory
    file_type: List[str] = field(default_factory=lambda: ["*.jpg", "*.png"])


def get_config() -> DemoCN:
    return AP(config_class=DemoCN).parse_args()


args = get_config()


from hamer.utils import SkeletonRenderer, recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

from .util import infer, init_detector, resize, stack_and_pad


import jax

class HttpServer:

    def __init__(self):

        # Download and load checkpoints
        download_models(CACHE_DIR_HAMER)
        self.model, self.model_cfg = load_hamer(args.checkpoint)

        pprint(self.model_cfg)

        # Setup HaMeR model
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.detector = init_detector()  # Load detector
        self.vitpose = ViTPoseModel(self.device)  # keypoint detector
        self.renderer = Renderer(
            self.model_cfg, faces=self.model.mano.faces
        )  # Setup the renderer
        self.skrenderer = SkeletonRenderer(self.model_cfg)

        # torch models dont compile... so we are done

    def run(self, port=8000, host="0.0.0.0"):
        self.app = FastAPI()
        self.app.post("/query")(self.forward)
        self.app.post("/reset")(self.reset)
        uvicorn.run(self.app, host=host, port=port)

    def reset(self, payload: Dict[Any, Any]):
        return "reset"

    def forward(self, payload: Dict[Any, Any]):
        try:

            obs = payload["observation"]
            for key in obs:
                if "image" in key:
                    obs[key] = resize(obs[key])

            out = infer(
                obs,
                self.model,
                self.model_cfg,
                self.detector,
                self.vitpose,
                self.renderer,
                self.skrenderer,
            )

            out = jax.tree.map(
                lambda x: x[0], out.data, is_leaf=is_leaf
            )  # everything is wrapped in list

            out = flatten(out)

            clean = lambda x: (
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            )
            out = jax.tree.map(clean, out)

            pprint(spec(out))



            return json_response(out)

        except Exception as e:
            print(traceback.format_exc())
            return "error"


def main():
    import argparse

    tf.config.set_visible_devices([], "GPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host to run on", default="0.0.0.0", type=str)
    parser.add_argument("--port", help="Port to run on", default=8000, type=int)
    args = parser.parse_args()

    server = HttpServer()
    server.run(args.port, args.host)


if __name__ == "__main__":
    main()
