import glob
from tqdm import tqdm
import json
import os
import os.path as osp
import time

import cv2
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def display(image):
    """Displays an image using OpenCV."""
    cv2.imshow("Camera 0", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(100)


def filter_episodes(dataset, path, filtered):
    _f = {"yes": [], "no": []}

    for i, ep in enumerate(dataset):

        f = filtered.get(path, {})
        if i in f.get("yes", []) or i in f.get("no", []):
            continue  # already in yes

        if len(ep["steps"]) > 7:
            _f["no"].append(i)

        for j, step in enumerate(ep["steps"]):

            imgs = step["observation"]["img"]
            imgs = jax.tree.map(lambda x: np.array(x), imgs)
            imgs = np.concatenate(list(imgs.values()), axis=1)

            display(imgs)

        # Ask the user if they want to keep this episode
        # if cv2 waitkey is y, keep the episode if n discard the episode
        if cv2.waitKey(0) == ord("y"):
            _f["yes"].append(i)
            print(f"Episode {i} kept.")
        else:
            _f["no"].append(i)
            print(f"Episode {i} discarded.")

    return _f


def save_to_tfds(episodes, output_dir, name="filtered_dataset"):
    """Saves the filtered episodes back into TFDS format."""
    # Create a new dataset builder for the filtered dataset
    builder = tfds.core.DatasetBuilder(
        name=name,
        data_dir=output_dir,
        features=tfds.features.FeaturesDict(
            {
                "img": tfds.features.FeaturesDict(
                    {
                        "camera_0": tfds.features.Tensor(
                            shape=(640, 640, 3), dtype=tf.uint8
                        ),
                        "camera_1": tfds.features.Tensor(
                            shape=(640, 640, 3), dtype=tf.uint8
                        ),
                        "wrist": tfds.features.Tensor(
                            shape=(640, 640, 3), dtype=tf.uint8
                        ),
                    }
                ),
                "robot": tfds.features.FeaturesDict(
                    {
                        "joints": tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                        "position": tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                    }
                ),
                "action_info": tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                "reward_info": tfds.features.Tensor(shape=(), dtype=tf.float64),
                "discount_info": tfds.features.Tensor(shape=(), dtype=tf.float64),
            }
        ),
        version="1.0.0",
    )

    # Define the output split (e.g., 'train')
    split_info = tfds.core.SplitGenerator(
        name="train",
        gen_kwargs={"examples": episodes},
    )

    # Create the new TFDS structure
    os.makedirs(output_dir, exist_ok=True)
    with tfds.core.utils.write_data_dir(output_dir) as data_dir:
        writer = tfds.core.Writer(builder.info, data_dir)

        # Write each filtered example back into the TFDS format
        for episode in episodes:
            writer.write(episode)

        writer.finalize()

    print(f"Filtered episodes saved to TFDS format at: {output_dir}")


def main():

    data_dir = osp.expanduser("~/data")
    env_name = "xgym-lift-v0"
    inpath = osp.join(data_dir, f"{env_name}-*")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    outpath = osp.join(data_dir, f"{env_name}-filtered-{timestr}")

    filterpath = osp.join(data_dir, "filtered.json")
    try:
        with open(filterpath, "r") as f:
            filtered = json.load(f)
    except Exception as e:
        print(f"Error loading filtered episodes: {e}")
        filtered = {}

    print(filtered)

    paths = list(glob.glob(inpath))
    for p in tqdm(paths):
        f = filtered.get(p, {})
        if f == "err":
            continue

        yes = f.get("yes", [])
        no = f.get("no", [])
        # yes must not include elements in no
        f["yes"] = list(set(yes) - set(no))

        try:
            dataset = tfds.builder_from_directory(p).as_dataset(split="all")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            filtered[p] = "err"
            continue

        print(p)

        _f = filter_episodes(dataset, p, filtered)

        if f == {}:
            f = _f
        else:
            f["yes"] = list(set(f["yes"] + _f["yes"]))
            f["no"] = list(set(f["no"] + _f["no"]))

        filtered[p] = f

        # save filtered to json
        print(filtered[p])
        with open(filterpath, "w") as f:
            json.dump(filtered, f)

    for p in paths:
        # remove dir where value is err
        if filtered[p] == "err":
            print(f"Removing {p}")
            os.system(f"rm -rf {p}")

    quit()

    # Save filtered episodes back to TFDS format
    save_to_tfds(filtered, outpath, name=env_name)


if __name__ == "__main__":
    main()
