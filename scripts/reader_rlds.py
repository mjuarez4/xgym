import cv2
import numpy as np
import xgym
import tensorflow_datasets as tfds

ds = tfds.load('xgym_lift_single')['train']

for ep in ds:
    for s in ep['steps']:
        # print(s)

        action = np.array(s['action']).tolist()
        action = np.array([round(a, 4) for a in action])
        print(action)

        img = np.concatenate(list(s['observation']['image'].values()), axis=1)

        cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(50)

    print()
