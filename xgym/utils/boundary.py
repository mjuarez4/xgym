import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

# from gello.robots.xarm_robot import RobotState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# @dataclass(frozen=True)
@dataclass()
class PartialRobotState:
    """Represents a partial state of the robot
    inspired by gello RobotState
    """

    cartesian: Optional[Union[List[float], np.ndarray]] = None
    gripper: Optional[Union[float, np.ndarray]] = None
    joints: Optional[Union[List[float], np.ndarray]] = None
    aa: Optional[Union[List[float], np.ndarray]] = None

    @staticmethod
    def from_vector(vector: np.ndarray) -> "PartialRobotState":
        """Converts a vector to a robot state object."""

        return PartialRobotState(
            cartesian=np.array(vector[:3]),
            aa=np.array(vector[3:6]),
            gripper=vector[6] if len(vector) == 7 else None,
        )

    def __add__(self, other: "PartialRobotState") -> "PartialRobotState":
        """assumes the second robot state is a delta"""
        logger.warn("double check for correctness adding aa")

        things = {}
        for key in self.__annotations__:
            v1 = getattr(self, key)
            v2 = getattr(other, key)

            if v1 is None or v2 is None:
                things[key] = None
            else:
                things[key] = v1 + v2

        return PartialRobotState(**things)


class Boundary(ABC):
    @abstractmethod
    def contains(self, state: PartialRobotState) -> bool:
        pass


@dataclass
class CartesianBoundary(Boundary):
    """Represents an axis-aligned bounding box in 3D space."""

    min: PartialRobotState
    max: PartialRobotState

    def post_init(self):
        assert len(self.min.cartesian) == 3
        assert len(self.max.cartesian) == 3

    def contains(self, state: PartialRobotState) -> bool:
        """Checks if the given point is inside the bounding box.

        Returns:
            bool: True if the point is inside the box, False otherwise.
        """

        logger.info(
            f"{self.min.cartesian} <= {state.cartesian} <= {self.max.cartesian}"
        )
        return np.logical_and(
            (self.min.cartesian <= state.cartesian),
            (state.cartesian <= self.max.cartesian),
        ).all()


@dataclass
class AngularBoundary(Boundary):

    min: PartialRobotState
    max: PartialRobotState

    def post_init(self):
        assert len(self.min.aa) == 3
        assert len(self.max.aa) == 3

    def contains(self, state: PartialRobotState) -> bool:
        """Checks if the given point is inside the bounding box."""
        clean = lambda x: np.where((x > np.pi) | (x < -np.pi), x % np.pi, x)
        logger.info(f"{self.min.aa} <= {clean(state.aa)} <= {self.max.aa}")
        return np.logical_and(
            (self.min.aa <= clean(state.aa)),
            (clean(state.aa) <= self.max.aa),
        ).all()


@dataclass
class GripperBoundary(Boundary):
    min: float
    max: float

    def contains(self, state: PartialRobotState) -> bool:
        """Checks if the given point is inside the bounding box."""
        if state.gripper is None:
            return True
        return self.min <= state.gripper <= self.max


@dataclass
class JointBoundary(Boundary):
    """Represents a joint limit in the robot's configuration space."""

    min: PartialRobotState
    max: PartialRobotState

    def post_init(self):
        assert len(self.min.joins) == 7
        assert len(self.max.joins) == 7

    def contains(self, state: PartialRobotState) -> bool:
        """Checks if the given robot state is within the joint limits.

        Returns:
            bool: True if the state is within the limits, False otherwise.
        """
        if not isinstance(state, PartialRobotState):
            raise ValueError("State must be a RobotState object.")

        return np.logical_and(
            (self.min.joints <= state.joints), (state.joints <= self.max.joints)
        ).all()


@dataclass
class CompositeBoundary(Boundary):
    bounds: List[Boundary]

    def add(self, b: Union[CartesianBoundary, JointBoundary]) -> None:
        self.bounds.append(b)

    def clear(self) -> None:
        self.bounds = []


class AND(CompositeBoundary):
    """logical AND operation between multiple boundaries."""

    def contains(self, state: PartialRobotState) -> bool:
        c = [b.contains(state) for b in self.bounds]
        logger.info(f"AND: {c}")
        return all(c)


class OR(CompositeBoundary):
    """logical OR operation between multiple boundaries."""

    def contains(self, state: PartialRobotState) -> bool:
        return any([b.contains(state) for b in self.bounds])


@dataclass
class NOT(Boundary):
    """logical NOT operation on a single boundary."""

    bound: Boundary

    def contains(self, state: PartialRobotState) -> bool:
        return not self.bound.contains(state)
