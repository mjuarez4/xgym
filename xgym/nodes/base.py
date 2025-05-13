import json
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool, Float32MultiArray, String
from xarm_msgs.msg import CIOState, RobotMsg


from functools import partial
from cv_bridge import CvBridge


class Base(Node):

    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info("initializing...")
        self.p = 0
        self.data = {}

        self.active = False

        self.active_pub = self.create_publisher(Bool, "/xgym/active", 10)
        self.active_sub = self.create_subscription(
            Bool, "/xgym/active", self.set_active, 10
        )

    def list_camera_topics(self):
        topics = self.get_topic_names_and_types()
        cams = [t for t, types in topics if "camera" in t]
        return cams

    def build_cam_subs(self):
        self.camqos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.cams = self.list_camera_topics()
        print(self.cams)
        self.data = self.data | {k: np.zeros((224, 224, 3)) for k in self.cams}

        self.subs = {
            k: self.create_subscription(
                CompressedImage, k, partial(self.set_image, key=k), self.camqos
            )
            for k in self.cams
        }

        # ats = ApproximateTimeSynchronizer(
        # list(self.subs.values()), queue_size=10, slop=0.1
        # )
        # ats.registerCallback(self.sync_callback)

        self.bridge = CvBridge()
        self.get_logger().info("Initialized Camera Subs.")

    def set_active(self, msg: Bool):
        self.active = msg.data
        self.p = 0
        # self.logp(f"setting active {self.active}")

    def set_image(self, msg, key):

        # if self.p % self.hz == 0:
        # self.get_logger().info(f"Received image from {key}")

        # frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # if it is compressed image,

        frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.data[key] = frame

    def set_period(self):
        def fn():
            self.p += 1

        self.period_timer = self.create_timer(1 / self.hz, fn)

    def logp(self, desc):
        if self.p % self.hz == 0:
            self.get_logger().info(desc)

    def timestamp(self):
        """Returns timestamp relative to node startup."""
        return round(time.time() - self.t0, 4)
