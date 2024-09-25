from dataclasses import dataclass
import numpy as np

@dataclass
class BoxBoundary:
    """
    Represents an axis-aligned bounding box in 3D space.
    """
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    def contains(self, point: np.ndarray) -> bool:
        """
        Checks if the given point is inside the bounding box.

        Args:
            point (np.ndarray): A 3D point as a numpy array [x, y, z].

        Returns:
            bool: True if the point is inside the box, False otherwise.
        """
        if point.ndim != 1 or point.shape[0] != 3:
            raise ValueError("Point must be a 1D numpy array with 3 elements.")

        x, y, z = point
        inside = (
            self.min_x <= x <= self.max_x and
            self.min_y <= y <= self.max_y and
            self.min_z <= z <= self.max_z
        )
        return inside


