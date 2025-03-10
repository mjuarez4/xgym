import threading

import numpy as np
import rclpy
from evdev import InputDevice, ecodes
from rclpy.node import Node
from std_msgs.msg import Bool, Int32MultiArray

from xgym.nodes.base import Base


class Governor(Base):

    def __init__(self):
        super().__init__("governor")

        self.pedal_sub = self.create_subscription(
            Int32MultiArray, "/xgym/pedal", self.on_pedal, 10
        )

        self.pubs = {
            "replay": self.create_publisher(Bool, "/xgym/gov/writer/replay", 10),
            "del": self.create_publisher(Bool, "/xgym/gov/writer/del", 10),
        }

    def on_pedal(self, msg):
        data = {
            "ss": msg.data[0],  # start stop
            "fn1": msg.data[1],
            "fn2": msg.data[2],
        }

        if data["ss"] == 1:
            self.active_pub.publish(Bool(data=not self.active))
            self.get_logger().info("XGym Reactivated")

        if data["fn1"] == 1:
            if self.active is False:  # TODO add toggle replay
                self.get_logger().info("Replay")
                self.pubs["replay"].publish(Bool(data=True))
        if data["fn1"] == 2:
            if self.active is False:
                self.pubs["del"].publish(Bool(data=True))


class ModelGovernor(Governor):

    def __init__(self):
        super().__init__("model_governor")
        pass


class ReplayGovernor(Governor):

    def __init__(self):
        super().__init__("model_governor")
        # TODO use space mouse to control video scrubbing ?
        pass
