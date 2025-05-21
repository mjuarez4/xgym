from dataclasses import dataclass
from functools import partial
from typing import List

from april import AprilDetection, Calibrator
import cv2
import numpy as np
from rich.pretty import pprint
from scipy.spatial.transform import Rotation as R
import tyro
from urdf.robot import RobotTree

from xgym import BASE

np.set_printoptions(suppress=True, precision=3)


@dataclass
class Config:
    pass
    # source: CameraConfig | FileConfig


# Helper function to invert a 4x4 homogeneous transformation matrix
def invert_transformation(T):
    """Inverts a 4x4 homogeneous transformation matrix."""
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.identity(4)
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, 3] = t_inv
    return T_inv


# Helper function to extract R (3x3) and t (3x1) from T (4x4)
def extract_Rt(T):
    """Extracts rotation matrix R and translation vector t from a 4x4 matrix T."""
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)  # Ensure t is a column vector
    return R, t


@dataclass
class RobotPose:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    @classmethod
    def from_vector(cls, vec: List[float]) -> "RobotPose":
        assert len(vec) == 6, "Vector must have 6 elements"
        return cls(*vec)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_matrix(self) -> np.ndarray:
        rot = R.from_euler("xyz", [self.rx, self.ry, self.rz], degrees=False)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = self.position
        return T


def make_motion_pairs(poses):
    """Turn absolute poses into relative motion pairs."""
    motions = []
    for i in range(len(poses)):
        for j in range(len(poses)):
            if i == j:
                continue
            A_ij = np.linalg.inv(poses[i]) @ poses[j]
            motions.append(A_ij)
    return motions


r = np.eye(4)
r[:3, :3] = np.array(
    [
        [0, 0, 1],  # new_x = old_z
        [1, 0, 0],  # new_y = old_x
        [0, 1, 0],  # new_z = old_y
    ]
)


# hardcode X
X_eef = np.eye(4)
X_eef[0, 3] = 0.0
X_eef[1, 3] = -0.06
X_eef[2, 3] = -0.03  # 3cm
# X_eef = X_eef @ np.linalg.inv(r)

X_tcp = np.eye(4)
X_tcp[0, 3] = 0.0
X_tcp[1, 3] = 0.172
X_tcp[2, 3] = 0.0

X = X_eef @ X_tcp @ np.linalg.inv(r)

rx_flip = np.array(
    [
        [-1, 0, 0, 0],  # -x → x
        [0, 1, 0, 0],  # y unchanged
        [0, 0, 1, 0],  # z unchanged
        [0, 0, 0, 1],  # homogeneous coordinate
    ]
)
ry_flip = np.array(
    [
        [1, 0, 0, 0],  # x unchanged
        [0, -1, 0, 0],  # -y → y
        [0, 0, 1, 0],  # z unchanged
        [0, 0, 0, 1],  # homogeneous coordinate
    ]
)


def main(cfg: Config):
    calibrator = Calibrator()
    # source = cfg.source.create()

    T_base_eef, T_camera_tag = [], []

    ways = np.load(BASE / "ways.npz")
    ways = {k: ways[k] for k in ways.files}

    rtree = RobotTree()

    kinv_path = BASE / "kinvs.npz"
    kinvs = np.load(kinv_path)["kinvs"] if kinv_path.exists() else None
    pprint(kinvs.shape)

    msg = "Kinvs not found... run xgym/calibrate/urdf/robot"
    assert kinvs is not None, msg

    cam2base_path = BASE / "base.npz"
    _bases = np.load(cam2base_path)["base"] if cam2base_path.exists() else None

    pprint(_bases)

    centers, frames, bases = [], [], []
    for i in range(len(ways["joints"])):
        d = {k: v[i] for k, v in ways.items()}
        kinv = kinvs[i] if kinvs is not None else None
        # pprint(source[0].spec(d))

        joints = d["joints"]
        _pose = d["poses"]
        frame = d["frames"]

        # frame = d["/xgym/camera/low"]
        # pose = d["xarm_pose"]
        _pose[:3] /= 1e3
        pose = RobotPose.from_vector(_pose)

        det: AprilDetection = calibrator.process(frame)

        """
        for a, b in zip(centers[:-1], centers[1:]):
            a, b = tuple(a.astype(int)), tuple(b.astype(int))
            cv2.circle(frame, a, 2, (0, 0, 255), -1)
            cv2.line(frame, a, b, (0, 0, 255), 1)
        """

        if det is not None and pose is not None:
            T_base_eef.append(pose.to_matrix())
            T_camera_tag.append(det.to_matrix())
            centers.append(det.center)
            frames.append(frame)

            AX = det.to_matrix() @ X
            calibrator.draw_pose_axes(
                frame,
                AX[:3, :3],
                AX[:3, 3],
            )

            kin: tf.Transform3d = rtree.set_pose(joints, end_only=True).get_matrix()
            kinv = np.linalg.inv(kin)[0] if kinv is not None else kinv

            kins = {
                k: v.get_matrix().numpy()[0] for k, v in rtree.set_pose(joints).items()
            }

            if kinv is not None:
                keypoints = {
                    k: np.array(AX) @ np.array(kinv) @ np.array(v)
                    for k, v in kins.items()
                }
                if _bases is not None:
                    keypoints = {k: _bases @ v for k, v in kins.items()}

                print(keypoints.keys())

            base = np.array(AX) @ np.array(kinv)  # @ rx_flip
            bases.append(base)
            if _bases is not None:
                base = _bases

            calibrator.draw_pose_axes(
                frame,
                base[:3, :3],
                base[:3, 3],
            )

            # pprint(base)
            points = []
            if kinv is not None:
                for k, v in keypoints.items():
                    point = calibrator.project_points(
                        (0, 0, 0),
                        v[:3, :3],
                        v[:3, 3],
                    )
                    points.append(point.reshape(-1))

            for a, b in zip(points[:-1], points[1:]):
                # print(a.shape, b.shape)
                pink = (255, 0, 255)
                a, b = tuple(a.astype(int)), tuple(b.astype(int))
                cv2.circle(frame, a, 4, pink, -1)
                cv2.line(frame, a, b, pink, 1)
            cv2.circle(frame, b, 4, pink, -1)

        # if len(T_base_eef) > 0:
        # tmp(T_camera_tag[-1], T_base_eef[-1], frame, calibrator)

        # Upscale for better visibility
        # h, w = frame.shape[:2]
        # scale = 4
        # frame = cv2.resize(frame.copy(), (w * scale, h * scale))
        cv2.imshow("AprilTag Detection", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(0) & 0xFF == ord("q"):
            quit()

    # save frames to mp4 video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (640, 480))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out.release()

    bases = np.array(bases).mean(axis=0)
    # pprint(bases)
    np.savez(BASE / "base.npz", base=bases)

    quit()

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    assert len(T_base_eef) == len(T_camera_tag), "Number of poses must match"
    assert len(T_base_eef) >= 3, "Need at least 3 poses for calibration"

    """
    A = make_motion_pairs(T_base_eef)
    B = make_motion_pairs(T_camera_tag)

    As,Bs = T_base_eef, T_camera_tag

    # cv2.calibrateCamera
    # cv2.calibrateHandEye
    # cv2.calibrateRobotWorldHandEye

    def extract_rt(T):
        R = T[:3, :3]
        t = T[:3, 3]
        return R, t

    quit()

    import solve
    X = solve.torch_handeye_solver(As, Bs)
    """

    # hardcode X
    # X = np.eye(4)
    # X[0, 3] = 0.0
    # X[1, 3] = -0.08  # -8cm
    # X[2, 3] = 0.03  # 3cm

    # X = X.reshape(1, *X.shape)
    rhs = np.array(T_camera_tag) @ X

    bases = rhs @ np.linalg.inv(T_base_eef)

    # base = np.linalg.inf(T_base_eef[0] )

    pprint(X)
    pprint(rhs)

    # Run Tsai hand-eye calibration
    # R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    # R_gripper2base,
    # t_gripper2base,
    # R_target2cam,
    # t_target2cam,
    # method=cv2.CALIB_HAND_EYE_TSAI,
    # )

    # Final 4x4 transform
    # T_cam2gripper = np.eye(4)
    # T_cam2gripper[:3, :3] = R_cam2gripper
    # T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

    # print("T_camera_eef:")
    # pprint(T_cam2gripper)

    # T_base_camera = T_base_eef[i] @ np.linalg.inv(T_camera_tag[i]) @ T_cam2gripper
    # print("T_base_camera:")
    # pprint(T_base_camera)

    project2im = partial(
        project_points_to_image,
        intr=calibrator.intr.mat,
        dist_coeffs=calibrator.intr.distortion,
    )

    # Example points
    eef_point = T_base_eef[0][:3, 3]  # (3,)
    base_point = np.array([0, 0, 0])  # base origin

    print("EEF Point:")
    pprint(eef_point)

    print("cam intrinsics")
    pprint(calibrator.intr.mat)

    # Convert base->cam
    # T_base_to_cam = np.linalg.inv(T_base_camera)

    # Stack and project
    points_world = np.stack([base_point, eef_point], axis=0)
    # image_pts = project2im(points_world, T_base_to_cam)

    image_pts = project2im(rhs[:, :3, 3], np.eye(4))

    bases = project2im(bases[:, :3, 3].reshape(-1, 3), np.eye(4))

    yellow = (255, 255, 0)
    black = (0, 0, 0)

    # Draw on image
    for pt, b, frame in zip(image_pts, bases, frames):
        x, y = int(pt[0]), int(pt[1])
        # print(x, y)
        print(b)
        cv2.circle(frame, (x, y), 6, yellow, -1)
        cv2.circle(frame, (int(b[0]), int(b[1])), 6, (255, 0, 255), -1)
        cv2.putText(
            frame,
            "Point",
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            black,
            2,
        )

        cv2.imshow("Projected Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main(tyro.cli(Config))
