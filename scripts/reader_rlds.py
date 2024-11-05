import cv2
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

import xgym

ds = tfds.load("xgym_lift_single")["train"]

for i, ep in tqdm(enumerate(ds)):

    imgs = []
    for s in ep["steps"]:
        # print(s)

        action = np.array(s["action"]).tolist()
        action = [round(a, 3) for a in action]
        print(action)

        img = np.concatenate(list(s["observation"]["image"].values()), axis=1)
        imgs.append(img)

        cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(5)

    # write video to gif with imageio
    import imageio

    imageio.mimsave(f"ep_{i}.gif", imgs, fps=10)

    # cv2.waitKey(0)
    # if i > 10:
        # break

    print()
