import abc
from typing import Any

import mujoco
import numpy as np


class MujocoSimObserver(abc.ABC):
    """
    Base class for observers that extract measurements from a running MujocoSim.

    Observers can maintain state across simulation steps but should reset that state
    when reset() is called at the start of each episode.
    """

    @abc.abstractmethod
    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any | dict[str, Any]:
        """
        Extract observation from the simulation state.

        Args:
            model: The MuJoCo model
            data: The MuJoCo simulation data

        Returns:
            The observed value (should always return a valid value for continuous data stream)
            or a dictionary of observed values
        """
        pass

    def reset(self):
        """
        Optional method to reset observer state for a new episode.

        Subclasses can override this method if they maintain state that needs to be
        cleared between episodes. Default implementation does nothing.
        """
        return None


class BodyDistance(MujocoSimObserver):
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

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> float | dict[str, float]:
        """
        Compute the distance between two body centers.

        Args:
            model: The MuJoCo model
            data: The MuJoCo simulation data

        Returns:
            The Euclidean distance between the two body centers.
        """
        pos_1 = data.body(self.body_name_1).xpos.copy()
        pos_2 = data.body(self.body_name_2).xpos.copy()
        distance = np.linalg.norm(pos_1 - pos_2)
        return float(distance)


def _soft_less(x: float, y: float, epsilon) -> float:
    return np.clip((y - x + epsilon / 2) / epsilon, 0.0, 1.0)


class StackingSuccess(MujocoSimObserver):
    """
    Observer that computes a success score for stacking green box on top of red box.

    Returns a score in [0, 1] where 1.0 indicates complete success.
    Success requires:
    - Red box remains stable on table
    - Green box is positioned above and resting on red box
    - Green box is stable (low velocity)
    - Gripper is clear of the boxes
    """

    def __init__(
        self,
        red_body_name: str,
        green_body_name: str,
        gripper_body_name: str,
        red_box_half_height: float = 0.01,
        green_box_half_height: float = 0.01,
        xy_tolerance: float = 0.025,
        z_stack_tolerance: float = 0.015,
        red_movement_tolerance: float = 0.04,
        gripper_clearance: float = 0.12,
        velocity_threshold: float = 0.02,
        table_height: float = 0.30,
        full_report: bool = False,
    ):
        """
        Initialize the observer.

        Args:
            red_body_name: Name of the red box body
            green_body_name: Name of the green box body
            gripper_body_name: Name of the gripper body
            red_box_half_height: Half-height of red box (m), from box size config
            green_box_half_height: Half-height of green box (m), from box size config
            xy_tolerance: Max XY distance for green box to be considered "above" red box (m)
            z_stack_tolerance: Tolerance for Z height difference for stacking (m)
            red_movement_tolerance: Max distance red box can move from initial position (m)
            gripper_clearance: Min distance gripper should be from boxes (m)
            velocity_threshold: Max velocity for green box to be considered stable (m/s)
            table_height: Height of table surface (m), computed as table pos Z + table thickness
            full_report: Whether to return a full report of all scores
        """
        self.red_body_name = red_body_name
        self.green_body_name = green_body_name
        self.gripper_body_name = gripper_body_name
        self.red_box_half_height = red_box_half_height
        self.green_box_half_height = green_box_half_height
        self.xy_tolerance = xy_tolerance
        self.z_stack_tolerance = z_stack_tolerance
        self.red_movement_tolerance = red_movement_tolerance
        self.gripper_clearance = gripper_clearance
        self.velocity_threshold = velocity_threshold
        self.table_height = table_height
        self.full_report = full_report

        self._red_initial_pos: np.ndarray | None = None

    def reset(self):
        """Reset the observer state for a new episode."""
        self._red_initial_pos = None

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        """Compute the stacking success scores."""

        red_pos = data.body(self.red_body_name).xpos.copy()
        green_pos = data.body(self.green_body_name).xpos.copy()
        gripper_pos = data.body(self.gripper_body_name).xpos.copy()

        # Get green box velocity
        green_body_id = model.body(self.green_body_name).id
        green_vel = np.linalg.norm(data.cvel[green_body_id][:3])

        # Initialize red box initial position on first call
        if self._red_initial_pos is None:
            self._red_initial_pos = red_pos.copy()

        # Check 1: Red box stability (hasn't moved much)
        red_movement = np.linalg.norm(red_pos[:2] - self._red_initial_pos[:2])
        red_stable = _soft_less(red_movement, self.red_movement_tolerance, self.red_movement_tolerance / 2)

        # Check 2: Red box is on table (Z position close to expected table surface + half-height)
        expected_red_z = self.table_height + self.red_box_half_height
        red_on_table = _soft_less(abs(red_pos[2] - expected_red_z), 0.05, 0.025)

        # Check 3: Green box XY alignment with red box
        xy_distance = np.linalg.norm(green_pos[:2] - red_pos[:2])
        xy_aligned = _soft_less(xy_distance, self.xy_tolerance, self.xy_tolerance / 2)

        # Check 4: Green box is stacked on red box (Z position)
        # Expected Z difference = red_box_half_height + green_box_half_height
        expected_z_diff = self.red_box_half_height + self.green_box_half_height
        z_diff = green_pos[2] - red_pos[2]
        z_stacked = _soft_less(abs(z_diff - expected_z_diff), self.z_stack_tolerance, self.z_stack_tolerance / 2)

        # Check 5: Green box is stable (low velocity)
        green_stable = _soft_less(green_vel, self.velocity_threshold, self.velocity_threshold / 2)

        # Check 6: Gripper is clear
        gripper_to_red = np.linalg.norm(gripper_pos - red_pos)
        gripper_to_green = np.linalg.norm(gripper_pos - green_pos)
        gripper_clear = _soft_less(
            self.gripper_clearance, min(gripper_to_red, gripper_to_green), self.gripper_clearance / 2
        )

        score = min(red_stable, red_on_table, xy_aligned, z_stacked, green_stable, gripper_clear)
        if self.full_report:
            scores = {}
            scores['.red_stable'] = red_stable
            scores['.red_on_table'] = red_on_table
            scores['.xy_aligned'] = xy_aligned
            scores['.z_stacked'] = z_stacked
            scores['.green_stable'] = green_stable
            scores['.gripper_clear'] = gripper_clear
            scores[''] = score
            return scores
        return score
