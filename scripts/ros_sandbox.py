import draccus
import os.path as osp
from dataclasses import dataclass, field
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import draccus
from xgym.controllers import SpaceMouseController
from xgym.utils.boundary import PartialRobotState as RS


import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

@dataclass
class RunCFG:

    task: str = input("Task: ").lower()
    base_dir: str = osp.expanduser("~/data")
    time: str = time.strftime("%Y%m%d-%H%M%S")
    env_name: str = f"xgym-sandbox-{task}-v0-{time}"
    data_dir: str = osp.join(base_dir, env_name)

    nsteps: int = 30 
    nepisodes: int = 100

class SandboxSubscriber(Node):
    """
    Manages episode execution and subscribes to controller & environment nodes.
    """
    def __init__(self, cfg):
        super().__init__('sandbox_subscriber')

        #ideally would declare all the parameters of dracuss
        self.declare_parameter("data_dir", cfg.data_dir)

        # ROS Subscriptions
        self.state_sub = self.create_subscription(Float32MultiArray, '/robot_state', self.state_callback, 10)
        self.command_sub = self.create_subscription(Float32MultiArray, '/robot_commands', self.command_callback, 10)

        # Reset service client
        self.reset_client = self.create_client(Empty, '/reset_environment')

        # Episode tracking
        self.n_episodes = n_episodes
        self.nsteps = nsteps
        self.freq = freq
        self.dt = 1 / freq
        self.episode_count = 0
        self.step_count = 0
        self.episode_active = True

        self.get_logger().info(f"Starting {n_episodes} episodes...")
        self.start_episode()

    def start_episode(self):
        """Requests environment reset."""
        self.get_logger().info(f"Starting episode {self.episode_count + 1}/{self.n_episodes}...")
        self.send_reset_request()

    def send_reset_request(self):
        """Resets the environment."""
        if self.reset_client.wait_for_service(timeout_sec=5.0):
            req = Empty.Request()
            self.reset_client.call_async(req)
        else:
            self.get_logger().error("Reset service unavailable.")

    def command_callback(self, msg):
        """Receives SpaceMouse commands (but does NOT publish them)."""
        self.get_logger().info(f"Received SpaceMouse command: {msg.data}")

    def state_callback(self, msg):
        """Logs the robot state updates from environment_node."""
        self.get_logger().info(f"Received robot state: {msg.data}")

@draccus.wrap()
def main(cfg: RunCFG):
    
    rclpy.init()
    node = SandboxSubscriber(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Sandbox Subscriber...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
