import time
from dataclasses import dataclass

import cv2
import jax
import numpy as np
import tyro
from rich.pretty import pprint
from webpolicy.deploy.client import WebsocketClientPolicy

from get_calibration import MyCamera

spec = lambda x: jax.tree.map(lambda arr: arr.shape, x)


@dataclass
class Config:
    host: str
    port: int


from mano_pipe_v3 import remap_keys, select_keys, solve_2d
import jax


def main(cfg: Config):
    pprint(cfg)

    client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)

    # cam = MyCamera(0)
    cap = cv2.VideoCapture(3)
    time.sleep(0.5)

    while True:
        ret, frame = cap.read()

        # frame = cv2.resize(frame, (224, 224))
        pack = {"img": frame}
        out = client.infer(pack)
        out = jax.tree.map(lambda x: x.copy(), out)

        out = remap_keys(out)
        out = select_keys(out)

        out = solve_2d(
            {
                "img": out["img"],
                "scaled_focal_length": out["scaled_focal_length"],
                "keypoints_3d": out["keypoints_3d"][0],
            }
        )

        pprint(spec(out))

        for point in out["keypoints_2d"].reshape(-1, 2):
            # convert from
            print(point)
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main(tyro.cli(Config))
