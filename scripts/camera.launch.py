from dataclasses import dataclass
import os.path as osp
import time

import draccus
import rclpy
from rclpy.executors import MultiThreadedExecutor

from xgym.nodes.camera import CameraNode
from xgym.nodes.viz import FastImageViewer

# import atexit
# import signal
# import cv2
# from xgym.gyms import Base, Lift, Stack
# from xgym.utils import boundary as bd
# from xgym.utils import camera as cu
# from xgym.controllers import (KeyboardController, ScriptedController, SpaceMouseController)


@dataclass
class RunCFG:
    # task: str = input("Task: ").lower()
    task: str = "demo"
    base_dir: str = osp.expanduser("~/data")
    time: str = time.strftime("%Y%m%d-%H%M%S")
    env_name: str = f"xgym-sandbox-{task}-v0-{time}"
    data_dir: str = osp.join(base_dir, env_name)
    nsteps: int = 30
    nepisodes: int = 100
    gello: bool = False


@draccus.wrap()
def main(cfg: RunCFG):
    """Main training loop with environment interaction."""
    # Start environment-related scripts
    rclpy.init()

    cameras = [
        CameraNode(idx=0, name="worm"),
        CameraNode(idx=10, name="side"),
        CameraNode(idx=8, name="over"),
        CameraNode(idx=6, name="rs"),
    ]

    # nodes = nodes | {x.name: x for x in cameras}
    nodes = {"viewer": FastImageViewer()}
    nodes = list(nodes.values())

    # import launch

    # bag = launch.actions.Node(
    # package="rosbag2_transport",
    # executable="record",
    # name="record",
    # )

    ex = MultiThreadedExecutor()
    for node in nodes:
        ex.add_node(node)
    ex.spin()

    # finally:
    # _ = [rclpy.spin(node) for node in nodes]
    # _ = [node.destroy_node() for node in nodes]
    # rclpy.shutdown()

    # for node in nodes:
    # node.destroy_node()

    rclpy.shutdown()
    quit()


if __name__ == "__main__":
    main()
