import os
import os.path as osp
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import numpy as np
from tqdm import tqdm
from xgym.utils import camera as cu


@dataclass
class RunCFG:

    auto_keep: bool = False
    path: str = None


@draccus.wrap()
def main(cfg: RunCFG):

    print(cfg)
    assert cfg.path is not None

    ds = [str(x) for x in Path(cfg.path).glob("*.npz")]

    for path in tqdm(ds):

        e = np.load(path, allow_pickle=True)
        e = {x: e[x] for x in e.files}

        for i in range(e["worm"].shape[0]):
            imgs = {k: e[k][i] for k, v in e.items()}
            imgs = cu.writekeys(imgs)
            imgs = np.concatenate(list(imgs.values()), axis=1)
            cv2.imshow("img", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
            # cv2.imshow("img", imgs)
            # cv2.waitKey(0) # key every frame/step
            cv2.waitKey(5)
            # cv2.waitKey(50)

        if cfg.auto_keep:
            continue

        # no autokeep so wait for user input
        if cv2.waitKey(0) != ord("y"):
            os.remove(path)
        else:
            pass


if __name__ == "__main__":
    main()
