import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from xgym.controllers import SpaceMouseController


class SpaceMouse(Node):
    """
    Reads input from the SpaceMouse and publishes to /robot_commands.
    """

    def __init__(self):
        super().__init__("controller_node")

        self.publisher = self.create_publisher(Float32MultiArray, "/robot_commands", 10)
        self.controller = SpaceMouseController()

        self.hz = 300
        self.timer = self.create_timer(1 / self.hz, self.publish_command)

        self.get_logger().info("Controller Node Initialized.")

    def publish_command(self):
        """Reads SpaceMouse input and publishes it."""

        action = self.controller.read()
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.publisher.publish(msg)
        # self.get_logger().info(f"Published SpaceMouse command: {action.round(2)}")


def main(args=None):
    rclpy.init(args=args)
    node = SpaceMouseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Controller Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
