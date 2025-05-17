import glob
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
import tyro
from rich.pretty import pprint
from tqdm import tqdm


def setup(cfg):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    col, row = cfg.board.col, cfg.board.row
    shape = (col, row)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
    objp *= cfg.board.mm  # Now objp is in mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # images = glob.glob('*.jpg')

    images = []
    cap = cv.VideoCapture(cfg.cam)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # cv.imshow('frame', frame)
        # img = cv.imread(fname)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, shape, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv.drawChessboardCorners(frame, shape, corners2, ret)

            key = cv.waitKey(1)
            if key == ord("s"):
                images.append(frame.copy())
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"Captured {len(images)} views")

        scale = 2
        cv.imshow(
            "img",
            cv.resize(
                frame,  # cv.flip(frame, 1),
                (frame.shape[1] * scale, frame.shape[0] * scale),
            ),
        )
        # cv.waitKey(500)

        key = cv.waitKey(1)
        if key == ord("q"):
            break

    # write them all to disk in npz
    np.savez("calibration.npz", objpoints=objpoints, imgpoints=imgpoints, images=images)

    cv.destroyAllWindows()
    cap.release()
    return objpoints, imgpoints, images


def calibration(objpoints, imgpoints, images):
    img = images[0]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return ret, mtx, dist, rvecs, tvecs


def undistortion(imgs, mtx, dist):
    """This is the easiest way. Just call the function and use ROI obtained above to crop the result."""

    img = imgs[0]
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), alpha=0, newImgSize=(w, h)
    )
    for img in tqdm(imgs):
        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        crop = lambda _im: _im[y : y + h, x : x + w]
        crop = lambda _im: _im
        dst = crop(dst)

        cv.imshow("undistorted", dst)
        cv.imshow("distorted", crop(img))
        cv.imshow("diff", cv.absdiff(dst, crop(img)))
        key = cv.waitKey(1000)
        if key == ord("q"):
            break


def reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))


INTR = np.array(
    [
        [956.42114345, 0.0, 643.61300833],
        [0.0, 955.93236333, 358.52128195],
        [0.0, 0.0, 1.0],
    ]
)


np.set_printoptions(suppress=True, precision=3)


@dataclass
class Board:

    mm: float  # square size in mm
    row: int
    col: int


@dataclass
class A4(Board):
    mm: float = 29.0
    row: int = 6
    col: int = 8


@dataclass
class IPhone(Board):
    mm: float = 9.0
    row: int = 6
    col: int = 9


@dataclass
class Config:

    board: A4 | IPhone = field(default_factory=lambda: A4())
    cam: int = 0


def main(cfg: Config):

    path_frames = Path(f"calibragion_cam{cfg.cam}.npz")
    path_mtx = Path(f"intr_mtx_cam{cfg.cam}.npy")

    if path_frames.exists():
        data = np.load(str(path_frames))
        objpoints = data["objpoints"]
        imgpoints = data["imgpoints"]
        images = data["images"]
    else:
        objpoints, imgpoints, images = setup(cfg)

    pprint(images[0].shape)

    ret, mtx, dist, rvecs, tvecs = calibration(objpoints, imgpoints, images)
    pprint(mtx)
    pprint(dist)
    np.save(path_mtx, mtx)
    undistortion(images, mtx, dist)
    reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)


if __name__ == "__main__":
    main(tyro.cli(Config))
