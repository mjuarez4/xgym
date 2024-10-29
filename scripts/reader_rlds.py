import cv2
import numpy as np
import xgym
import tensorflow_datasets as tfds

ds = tfds.load('xgym_lift_single')['train']

for ep in ds:
    for s in ep['steps']:
        print(s)

        img = np.array(s['observation']['image']['camera_0'])
        print(img)

        cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(500)
