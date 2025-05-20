import os.path as osp
import time
from dataclasses import dataclass, field

import draccus
import rclpy
import tyro
from rclpy.executors import MultiThreadedExecutor
from rich.pretty import pprint
from tqdm import tqdm

from xgym.nodes import (Camera, FootPedal, Gello, Governor, Model, SpaceMouse,Heleo,
                        Writer, Xarm)
from xgym.nodes.model import NOMODEL, ModelClientConfig
from xgym.nodes.robot import ControlMode, InputMode, RobotConfig


def default(x):
    return field(default_factory=lambda: x)


from typing import Optional


@dataclass
class RunCFG:

    task: str = "demo"
    dir: str = "."  # data directory

    seconds: int = 15
    nepisodes: int = 100

    input: InputMode = InputMode.GELLO
    ctrl: ControlMode = ControlMode.JOINT

    # TODO make a factory bc task
    model: ModelClientConfig = default(NOMODEL)

    @property
    def robot(self):
        return RobotConfig(input=self.input, ctrl=self.ctrl)

        """
        base_dir: str = osp.expanduser("~/data")
        time: str = time.strftime("%Y%m%d-%H%M%S")
        env_name: str = f"xgym-sandbox-{task}-v0-{time}"
        data_dir: str = osp.join(base_dir, env_name)
        """


@draccus.wrap()
def main(cfg: RunCFG):
    """Main training loop with environment interaction."""

    pprint(cfg)

    # Start environment-related scripts
    rclpy.init()

    cameras = [
        Camera(idx=0, name="low"),
        Camera(idx=10, name="side"),
        # Camera(idx=3, name="high"),
        Camera(idx=7, name="wrist"),
    ]
    cameras = {x.name: x for x in cameras}

    match cfg.input:
        case InputMode.GELLO:
            ctrl = Gello()
        case InputMode.SPACEMOUSE:
            ctrl = SpaceMouse()
        case InputMode.MODEL:
            ctrl = Model(cfg.model)
        case InputMode.HELEO:
            ctrl = Heleo(cfg.model)

    nodes = {
        "robot": Xarm(cfg.robot),
        # 'viewer': FastImageViewer(active=False),
        "ctrl": ctrl,
        "writer": Writer(
            seconds=cfg.seconds, dir=cfg.dir
        ),  # writer spins after cameras init
        "pedal": FootPedal(),
        "gov": Governor(),
    }
    nodes = cameras | nodes  # cameras spin before writer
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
    main(tyro.cli(RunCFG))
