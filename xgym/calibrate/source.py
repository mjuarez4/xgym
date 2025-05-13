from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import cv2
import jax
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm

# from registry import REG


# REG.endpoint()
class FrameSource(ABC):
    def __iter__(self) -> Iterator[np.ndarray]:
        """Returns an iterator over the frames."""
        return self

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of frames."""
        ...

    @abstractmethod
    def __next__(self) -> np.ndarray:
        """Returns the next frame, or raises StopIteration when done."""
        ...

    @staticmethod
    def spec(arr):
        return jax.tree.map(
            lambda x: x.shape if isinstance(x, np.ndarray) else type(x), arr
        )


@dataclass
class CameraConfig:
    index: int = 0
    # width: int = 1280
    # height: int = 720
    # fps: int = 30

    def create(self):
        return Camera(self.index)


# @REG.register("camera")
class Camera(FrameSource):

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def __next__(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return next(self)
        return frame

    def release(self):
        self.cap.release()

    def __del__(self):
        # graceful shutdown if not released manually
        if hasattr(self, "cap") and self.cap.isOpened():
            self.release()


from dataclasses import field


@dataclass
class FileConfig:
    dir: str
    stride: int = 3  # only read every nth frame
    limit: Optional[int] = None  # max number of frames to read
    idxs: Optional[list[int]] = field(default_factory=lambda: [0, 2])

    def create(self):
        return [
            DatReplay(
                self.dir,
                self.limit,
                self.stride,
                i,
            )
            for i in self.idxs
        ]


# @REG.register("dat")
class DatReplay(FrameSource):

    def __init__(self, dir: str, limit: Optional[int] = None, stride: int = 3, idx=0):
        from xgym.viz.memmap import read

        self.dir = dir
        self.files = list(Path(dir).rglob("*.dat"))
        pprint(self.files)
        assert self.files, f"No .dat files found in {dir}"

        self.info, self.data = read(self.files[idx])
        self.indices = range(0, len(self.data["time"]), stride)
        if limit:
            self.indices = list(self.indices)[:limit]
        self._iter = iter(self.indices)

    def __len__(self) -> int:
        """Get the number of frames."""
        return len(self.indices)

    def __getitem__(self, index: int) -> np.ndarray:
        """Get a specific frame by index."""
        if index >= len(self.data["time"]):
            raise IndexError("Index out of range")
        return jax.tree.map(lambda x: x[index], self.data)

    def __next__(self) -> Optional[np.ndarray]:
        i = next(self._iter)

        step = self[i]
        return step

        # try:
        # except StopIteration:
        # return None

        frame = step["/xgym/camera/low"]

        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame
