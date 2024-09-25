from typing import List
import numpy as np
from box_boundary import BoxBoundary

class BoundaryManager:
    """
    Manages multiple BoxBoundary instances and checks if a point is within any such box.
    """

    def __init__(self):
        self.boxes: List[BoxBoundary] = []

    def add_box(self, box: BoxBoundary) -> None:
        """
        Adds a new bounding box to the manager.

        Args:
            box (BoxBoundary): The bounding box to add.
        """
        self.boxes.append(box)

    def is_inside(self, point: np.ndarray) -> bool:
        """
        Checks if the point is inside any of the bounding boxes.

        Args:
            point (np.ndarray): A 3D point as a numpy array [x, y, z].

        Returns:
            bool: True if inside at least one box, False otherwise.
        """
        for box in self.boxes:
            if box.contains(point):
                return True
        return False

    def clear_boxes(self) -> None:
        """
        Clears all bounding boxes from the manager.
        """
        self.boxes = []

