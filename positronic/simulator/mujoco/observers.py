from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

# Type alias for observer callables
MujocoSimObserver = Callable[[mujoco.MjModel, mujoco.MjData], Any]


class BodyDistance:
    """
    Observer that computes the Euclidean distance between two body centers.
    """

    def __init__(self, body_name_1: str, body_name_2: str):
        """
        Initialize the observer.

        Args:
            body_name_1: Name of the first body
            body_name_2: Name of the second body
        """
        self.body_name_1 = body_name_1
        self.body_name_2 = body_name_2

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> float | None:
        """
        Compute the distance between two body centers.

        Args:
            model: The MuJoCo model
            data: The MuJoCo simulation data

        Returns:
            The Euclidean distance between the two body centers, or None if either body doesn't exist
        """
        try:
            pos_1 = data.body(self.body_name_1).xpos.copy()
            pos_2 = data.body(self.body_name_2).xpos.copy()
            distance = np.linalg.norm(pos_1 - pos_2)
            return float(distance)
        except KeyError:
            return None
