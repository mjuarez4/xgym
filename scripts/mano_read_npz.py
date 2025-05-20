import os
import os.path as osp
import sys
import numpy as np
import cv2
from tqdm import tqdm
from xgym.utils import camera as cu
#@dataclass
#class RunCFG:

#    base_dir: str = osp.expanduser("~/data")
#    env_name: str = "xgym-lift-v0"
#    data_dir: str = osp.join(base_dir, env_name)


#cfg = RunCFG()

import sys

#print(cfg.data_dir)


ds = [x for x in os.listdir(sys.argv[1]) if x.endswith(".npz")]

A = []
N = 0

from tqdm import tqdm

from pathlib import Path
folder = Path(sys.argv[1])

for e in tqdm(ds):

    path = str(folder / e)
    e = np.load(path, allow_pickle=True)
    e = {x: e[x] for x in e.files}
    
    # print([ v.shape for v in e.values()])
    idxs = []

    for i in range(e['worm'].shape[0]):
        imgs = {k:e[k][i] for k,v in e.items()}
        imgs = cu.writekeys(imgs)
        imgs = np.concatenate(list(imgs.values()), axis=1)
        cv2.imshow("img", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))
        # cv2.imshow("img", imgs)
        # cv2.waitKey(0) # key every frame/step
        cv2.waitKey(5)
        # cv2.waitKey(50)

    if cv2.waitKey(0) != ord("y"):
        os.remove(path)
    else:
        pass
