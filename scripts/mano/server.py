import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import jax
import numpy as np
import torch
import tyro
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import DEFAULT_CHECKPOINT, HAMER, MANO, download_models, load_hamer
from hamer.utils import SkeletonRenderer, recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from jax import numpy as jnp
from rich.pretty import pprint

from array_util import stack_and_pad
from util import infer, init_detector

# from vitpose_model import ViTPoseModel
from webpolicy.deploy.base_policy import BasePolicy
from webpolicy.deploy.server import WebsocketPolicyServer as Server


def resize(img, size=(224, 224)):
    img = tf.image.resize(img, size=size, method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


HAMER_STAT = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def unnormalize(img):
    """bring image back to 0-255 range"""
    img = img * np.array(HAMER_STAT["std"]) + np.array(HAMER_STAT["mean"])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


class Policy(BasePolicy):

    def __init__(self, cfg):
        self.cfg = cfg

        # Download and load checkpoints
        download_models(CACHE_DIR_HAMER)
        self.model, self.model_cfg = load_hamer(cfg.checkpoint)

        pprint(self.model_cfg)

        # Setup HaMeR model
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.detector = init_detector(cfg)  # Load detector
        self.vitpose = ViTPoseModel(self.device)  # keypoint detector
        self.renderer = Renderer(
            self.model_cfg, faces=self.model.mano.faces
        )  # Setup the renderer
        self.skrenderer = SkeletonRenderer(self.model_cfg)

        print("policy init done")
        # torch models dont compile... so we are done

    def reset(self):
        pass

    def infer(self, obs: dict):
        try:

            img = obs["img"]
            out = infer(
                i=0,
                img=img,
                detector=self.detector,
                vitpose=self.vitpose,
                device=self.device,
                model=self.model,
                model_cfg=self.model_cfg,
                renderer=self.renderer,
                args=self.cfg,
            )

            # everything is wrapped in list
            spec = lambda arr: jax.tree.map(lambda x: (type(x), x.shape), arr)
            is_leaf = lambda x: isinstance(x, (list, tuple))
            out = jax.tree.map(lambda x: x[0], out.data, is_leaf=is_leaf)
            out = flatten(out)

            clean = lambda x: (
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            )
            out = jax.tree.map(clean, out)

            pprint(spec(out))

            prepare = lambda x: cv2.resize(x.transpose(1, 2, 0), (224, 224))
            out["img_wrist"] = np.stack(prepare(x) for x in out.pop("img"))
            out["img_wrist"] = unnormalize(out["img_wrist"])
            out["img"] = img

            return out

        except Exception as e:
            print(traceback.format_exc())
            return "error"


@dataclass
class PolicyConfig:

    checkpoint: str = DEFAULT_CHECKPOINT  # Path to pretrained model checkpoint
    img_folder: str = "example_data"  # Folder with input images
    out_folder: str = "out_demo"  # Output folder to save rendered results
    side_view: bool = False  # If set, render side view also
    full_frame: bool = True  # If set, render all people together also
    save_mesh: bool = False  # If set, save meshes to disk also
    batch_size: int = 1  # Batch size for inference/fitting
    rescale_factor: float = 2.0  # Factor for padding the bbox
    body_detector: str = "vitdet"  # Using regnety improves runtime and reduces memory
    file_type: List[str] = field(default_factory=lambda: ["*.jpg", "*.png"])
    device: int = 0  # Cuda device to run the server on


@dataclass
class Config:
    """config node for demo"""

    policy: PolicyConfig = field(default_factory=PolicyConfig)

    port: int = 8002  # Port to run the server on
    host: str = "0.0.0.0"  # Host to run the server on


def main(cfg: Config):

    pprint(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.policy.device)

    policy = Policy(cfg.policy)
    server = Server(
        policy,
        host=cfg.host,
        port=cfg.port,
        metadata=None,
    )
    print("serving on", cfg.host, cfg.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Config))
