from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detection, Detector


@dataclass
class Intrinsics:
    fx: float  # focal length in pixels
    fy: float  # focal length in pixels
    cx: float
    cy: float

    distortion: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(5))

    def create(self):
        return (self.fx, self.fy, self.cx, self.cy)

    @property
    def mat(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )


@dataclass
class Distortion:
    k1: float  # radial distortion
    k2: float  # radial distortion
    p1: float  # tangential distortion
    p2: float  # tangential distortion
    k3: float  # radial distortion

    def from_vector(self, vec: List[float]) -> "Distortion":
        assert len(vec) == 5, "Vector must have 5 elements"
        return Distortion(*vec)

    def create(self):
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)


# Default intrinsics (approximate for 720p MacBook webcam)
INTR_MAC = Intrinsics(fx=600.0, fy=600.0, cx=640 / 2, cy=480 / 2)

INTR_LOGITECH_LOW = Intrinsics(
    fx=517.24991827,
    fy=517.94052204,
    cx=323.61016625,
    cy=252.13758151,
    distortion=np.array([[0.075, -0.124, 0.0, -0.0, -0.081]]),
)
INTR_LOGITECH_3 = Intrinsics(
    fx=522.777,
    fy=522.152,
    cx=322.813,
    cy=242.28,
    distortion=np.array([[0.079, -0.083, -0.0, -0.003, -0.18]]),
)

TAG3IN = 0.0762  # meters (3 inches)
TAG6IN = TAG3IN * 2
TAG9IN = TAG3IN * 3


import numpy as np
from scipy.spatial.transform import Rotation as R


def eef_pose_to_matrix(x, y, z, rx, ry, rz):
    rot = R.from_euler("zyx", [rz, ry, rx])  # ZYX means apply X first, then Y, then Z
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


@dataclass
class AprilDetection:
    """Detection with stronger types and utilities"""

    id: int
    tvec: np.ndarray
    pose_R: np.ndarray  # 3×3 rotation matrix
    pose_t: np.ndarray  # 3×1 translation vector

    corners: np.ndarray  # 4x2 array of corners
    center: np.ndarray  # 2x1 array of center

    @property
    def tag_id(self) -> int:
        # backwards compatibility
        return self.id

    @classmethod
    def from_detection(cls, det: Detection) -> "AprilDetection":
        return cls(
            id=det.tag_id,
            tvec=det.pose_t,
            pose_R=det.pose_R,
            pose_t=det.pose_t,
            corners=det.corners,
            center=det.center,
        )

    def to_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.pose_R
        T[:3, 3] = self.pose_t.flatten()
        return T


# Create AprilTag detector with your specified parameters
class Calibrator:

    def __init__(self, intr: Intrinsics = INTR_LOGITECH_LOW, tag_size: float = TAG3IN):

        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self.intr = intr
        self.tag_size = tag_size

        self.hist = []

    def process(self, frame: np.ndarray):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect tags with pose estimation
        dets: list[Detection] = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.intr.create(),
            tag_size=self.tag_size,
        )

        det = AprilDetection.from_detection(dets[0]) if dets else None
        self.hist.append(det)
        if not det:
            return

        tvec = det.pose_t  # shape (3, 1)
        position = tvec.flatten()  # shape (3,)

        # Convert rotation matrix to Rodrigues rotation vector
        rvec, _ = cv2.Rodrigues(det.pose_R)
        tvec = det.pose_t

        self.draw_around_tag(frame, det)
        # self.draw_pose_axes(frame, rvec, tvec)
        return det

    def draw_around_tag(self, frame, det: Detection):

        # draw box
        corners = det.corners.astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # draw center, and ID
        center = tuple(det.center.astype(int))
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
        return
        cv2.putText(
            frame,
            f"ID {det.tag_id}",
            (center[0] + 10, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    def project_points(self, points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray):
        """
        Project 3D points to 2D image plane using camera intrinsics and pose.
        :param points: 3D points to project (shape: Nx3)
        :param rvec: Rotation vector (shape: 3x1)
        :param tvec: Translation vector (shape: 3x1)
        :return: Projected 2D points (shape: Nx2)
        """
        # Project points
        points_2d, _ = cv2.projectPoints(
            points, rvec, tvec, self.intr.mat, self.intr.distortion
        )
        return points_2d

    def draw_pose_axes(
        self,
        frame,
        rvec,
        tvec,
        length=0.1,
        origin=np.zeros((1, 3), dtype=np.float32),
        offset2d=(0, 0),
    ):

        # Draw XYZ axes
        axes = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(
            -1, 3
        )
        axes = axes + origin  # Translate axes to the origin
        # origin = np.zeros((1, 3), dtype=np.float32)

        from rich.pretty import pprint

        pprint(tvec.shape)

        # Project 3D axes points to 2D image
        points_2d, _ = cv2.projectPoints(
            np.vstack((origin, axes)), rvec, tvec, self.intr.mat, self.intr.distortion
        )
        o, x, y, z = [tuple(p.ravel().astype(int)) for p in points_2d]

        o, x, y, z = [
            (int(p[0] + offset2d[0]), int(p[1] + offset2d[1])) for p in [o, x, y, z]
        ]

        pprint((o, x, y, z))

        plots = {
            "x": [x, red := (255, 0, 0)],
            "y": [y, green := (0, 255, 0)],
            "z": [z, blue := (0, 0, 255)],
        }

        # place an cross at the origin
        cv2.drawMarker(
            frame, o, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10
        )

        # Draw the axes
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.line(frame, o, x, red, 2)  # X axis - Red
        cv2.putText(frame, "X", (x[0] + 10, x[1]), font, 0.5, red, 2)
        cv2.line(frame, o, y, green, 2)
        cv2.putText(frame, "Y", (y[0] + 10, y[1]), font, 0.5, green, 2)
        cv2.line(frame, o, z, blue, 2)
        cv2.putText(frame, "Z", (z[0] + 10, z[1]), font, 0.5, blue, 2)
