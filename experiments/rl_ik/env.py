from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RobotState:
    """Represents the state of a 2-joint robotic arm

    Attributes:
        thetas: Joint angles in radians [theta1, theta2]
        omegas: Joint angular velocities [omega1, omega2]
        ee_pos: End effector position [x, y]
    """
    thetas: np.ndarray  # [2,]
    omegas: np.ndarray  # [2,]
    ee_pos: np.ndarray  # [2,]


class RoboticArmEnv(gym.Env):
    """
    2D Robotic Arm Environment with 2 joints.

    The arm uses position control - actions specify target joint angles.
    The arm follows a trajectory specified as sequence of (dx, dy) deltas.

    State space:
        - Current end effector position (x, y)
        - Current joint angles (theta1, theta2)
        - Current joint velocities (omega1, omega2)
        - Next target deltas (dx, dy) for steps_ahead steps

    Action space:
        - Target joint angles [theta1, theta2]

    Reward:
        Negative distance between current and expected end effector position.
        Expected position is cumulative sum of trajectory deltas.

    Episode ends when:
        - No more trajectory deltas remain

    Args:
        link_lengths: Lengths of arm segments [l1, l2]. Default [1.0, 1.0]
        dt: Simulation timestep. Default 0.01
        max_velocity: Maximum joint velocity in rad/s. Default 2π (1 rotation/sec)
    """

    def __init__(self,
                 link_lengths: Sequence[float] = (1.0, 1.0),
                 dt: float = 0.01,
                 max_velocity: float = 2 * np.pi,
                 steps_ahead: int = 1):

        self.L = np.array(link_lengths)
        self.dt = dt
        self.max_velocity = max_velocity

        # Action space: target joint angles (very large but finite range)
        self.action_space = gym.spaces.Box(
            low=np.array([-4.0 * np.pi, -4.0 * np.pi]),  # 4 full rotations
            high=np.array([4.0 * np.pi, 4.0 * np.pi]),
            dtype=np.float32)

        # Observation space: current state and target deltas
        self.steps_ahead = steps_ahead
        self.observation_space = gym.spaces.Box(
            low=np.array([-100.0 * np.pi] * 2 + [-10.0] * (4 + 2 * steps_ahead)),  # angles + other states
            high=np.array([100.0 * np.pi] * 2 + [10.0] * (4 + 2 * steps_ahead)),
            dtype=np.float32)
        self.trajectory: deque = deque()
        self.state: Optional[RobotState] = None
        self.target_pos: Optional[np.ndarray] = None

    def push_trajectory(self, point: Tuple[float, float]):
        """Push new trajectory point (x, y) to the end"""
        self.trajectory.append(np.array([point[0], point[1]]))

    def extend_trajectory(self, points: Sequence[Tuple[float, float]]):
        """Extend trajectory with new points"""
        for point in points:
            self.push_trajectory(point)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Initialize state
        self.state = RobotState(thetas=np.zeros(2), omegas=np.zeros(2), ee_pos=self._forward_kinematics(np.zeros(2)))

        # Initialize target as first trajectory point
        self.target_pos = self.trajectory[0]

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step"""
        # Update dynamics
        new_state = self._dynamics_step(self.state, action)
        self.state = new_state

        # Calculate reward based only on position error
        pos_error = self.target_pos - self.state.ee_pos
        reward = -np.linalg.norm(pos_error)

        # Update trajectory
        terminated = len(self.trajectory) <= self.steps_ahead
        truncated = False
        info = {}  # Add any debug info here if needed

        if not terminated:
            self.trajectory.popleft()
            self.target_pos = self.trajectory[0]

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Get current state including joint angles and velocities
        current_state = np.array([
            *self.state.ee_pos,  # end effector position (2)
            *self.state.thetas,  # joint angles (2)
            *self.state.omegas  # joint velocities (2)
        ])

        # Get position errors for upcoming trajectory points
        error_deltas = []
        for i in range(min(self.steps_ahead, len(self.trajectory))):
            point = self.trajectory[i]
            pos_error = point - self.state.ee_pos
            error_deltas.extend([pos_error[0], pos_error[1]])

        # Pad with zeros if needed
        padding_needed = self.steps_ahead - len(self.trajectory)
        if padding_needed > 0:
            error_deltas.extend([0.0] * (2 * padding_needed))

        return np.concatenate([current_state, error_deltas])

    def _forward_kinematics(self, thetas: np.ndarray) -> np.ndarray:
        """Calculate end effector position from joint angles"""
        theta1, theta2 = thetas
        x = self.L[0] * np.cos(theta1) + self.L[1] * np.cos(theta1 + theta2)
        y = self.L[0] * np.sin(theta1) + self.L[1] * np.sin(theta1 + theta2)
        return np.array([x, y])

    def _dynamics_step(self, state: RobotState, target_angles: np.ndarray) -> RobotState:
        """Execute one dynamics step using position control"""
        # Calculate desired angular velocity
        angle_error = target_angles - state.thetas
        desired_velocity = angle_error / self.dt

        # Clip to max velocity
        new_omegas = np.clip(desired_velocity, -self.max_velocity, self.max_velocity)

        # Update angles using current velocity
        new_thetas = state.thetas + new_omegas * self.dt

        # Update end effector position
        new_ee_pos = self._forward_kinematics(new_thetas)

        return RobotState(thetas=new_thetas, omegas=new_omegas, ee_pos=new_ee_pos)

    def render(self):
        """Render the environment using matplotlib"""
        if not hasattr(self, 'fig'):
            # Create figure with white background
            self.fig = plt.figure(figsize=(6, 6), facecolor='white')
            self.ax = self.fig.add_subplot(111)

        self.ax.clear()
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('white')

        # Draw future trajectory points
        if len(self.trajectory) > 0:
            trajectory_points = list(self.trajectory)
            trajectory_points = np.array(trajectory_points)
            self.ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'gray', alpha=0.5, linewidth=2)

        # Draw robot arm
        # Base joint
        base_pos = np.array([0., 0.])

        # First link
        joint_pos = self.L[0] * np.array([np.cos(self.state.thetas[0]), np.sin(self.state.thetas[0])])

        # Second link
        ee_pos = joint_pos + self.L[1] * np.array(
            [np.cos(self.state.thetas[0] + self.state.thetas[1]),
             np.sin(self.state.thetas[0] + self.state.thetas[1])])

        # Draw links
        self.ax.plot([base_pos[0], joint_pos[0]], [base_pos[1], joint_pos[1]], 'b-', linewidth=3, label='Link 1')
        self.ax.plot([joint_pos[0], ee_pos[0]], [joint_pos[1], ee_pos[1]], 'b-', linewidth=3, label='Link 2')

        # Draw joints
        self.ax.plot(base_pos[0], base_pos[1], 'ko', markersize=8)
        self.ax.plot(joint_pos[0], joint_pos[1], 'ko', markersize=8)
        self.ax.plot(ee_pos[0], ee_pos[1], 'ko', markersize=8)

        # Draw current target
        if self.target_pos is not None:
            self.ax.plot(self.target_pos[0], self.target_pos[1], 'ro', markersize=8)

        # Remove axes for cleaner visualization
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.fig.canvas.draw()
        # Convert canvas to image
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3, ))

        return img

    def close(self):
        """Clean up matplotlib resources"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            del self.fig
            del self.ax


def generate_smooth_trajectories(
        start_positions: np.ndarray,  # Shape: (N, 2)
        end_positions: np.ndarray,  # Shape: (N, 2)
        duration_sec: float,
        dt: float = 0.01,
        noise_std: float = 0.01,
        curviness: float = 1.0) -> list[list[tuple[float, float]]]:
    """Generate multiple smooth, human-like curved trajectories between points.

    Args:
        start_positions: Array of starting positions, shape (N, 2)
        end_positions: Array of ending positions, shape (N, 2)
        duration_sec: Movement duration in seconds
        dt: Time step in seconds
        noise_std: Standard deviation of noise to add for natural variation
        curviness: Amount of curve deviation (0.0 to 1.0)

    Returns:
        List of N trajectories, each containing (x, y) tuples representing absolute positions
    """
    # Ensure inputs are properly shaped 2D arrays
    start_positions = np.atleast_2d(start_positions)
    end_positions = np.atleast_2d(end_positions)

    num_trajectories = len(start_positions)
    times = np.arange(0, duration_sec + dt, dt)  # Include end point

    # Calculate directions and distances
    directions = end_positions - start_positions  # Shape: (N, 2)
    distances = np.linalg.norm(directions, axis=1, keepdims=True)  # Shape: (N, 1)
    distances = np.maximum(distances, 1e-6)  # Avoid division by zero

    # Calculate perpendicular vectors
    perpendiculars = np.column_stack([-directions[:, 1], directions[:, 0]]) / distances

    # Generate random offsets for control points
    random_offsets = np.random.normal(0, 0.2, (num_trajectories, 1))

    # Calculate control points
    midpoints = (start_positions + end_positions) / 2
    control_points = midpoints + perpendiculars * (distances * curviness * random_offsets)

    # Generate Bézier curve points
    tau = times / duration_sec  # Shape: (T,)
    tau_expanded = tau[:, np.newaxis, np.newaxis]  # Shape: (T, 1, 1)
    start_expanded = start_positions[np.newaxis, :, :]  # Shape: (1, N, 2)
    control_expanded = control_points[np.newaxis, :, :]  # Shape: (1, N, 2)
    end_expanded = end_positions[np.newaxis, :, :]  # Shape: (1, N, 2)

    # Generate positions using vectorized Bézier formula
    positions = ((1 - tau_expanded)**2 * start_expanded + 2 * (1 - tau_expanded) * tau_expanded * control_expanded +
                 tau_expanded**2 * end_expanded)  # Shape: (T, N, 2)

    # Add human-like variations
    variation_weight = 4 * tau * (1 - tau)  # Shape: (T,)
    noise = np.random.normal(0, noise_std, positions.shape)
    positions += noise * variation_weight[:, np.newaxis, np.newaxis]

    # Convert to list of trajectories with absolute positions
    trajectories = []
    for i in range(num_trajectories):
        trajectory = [(x, y) for (x, y) in positions[:, i]]
        trajectories.append(trajectory)

    return trajectories


class RandomTrajectoryEnv(gym.Wrapper):
    """Wrapper that generates random trajectories on reset"""

    def __init__(self, env: RoboticArmEnv, min_duration: float = 1.0, max_duration: float = 3.0):
        super().__init__(env)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.dt = env.dt

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        # Generate random start and end positions
        start_pos = self.env._forward_kinematics(np.zeros(2))  # Start from initial position
        end_pos = np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)])

        # Random duration
        duration = np.random.uniform(self.min_duration, self.max_duration)

        # Generate trajectory
        trajectories = generate_smooth_trajectories(start_positions=start_pos.reshape(1, 2),
                                                    end_positions=end_pos.reshape(1, 2),
                                                    duration_sec=duration,
                                                    dt=self.dt,
                                                    noise_std=0.01,
                                                    curviness=0.5)

        # Clear existing trajectory and add new one
        self.env.trajectory.clear()
        self.env.extend_trajectory(trajectories[0])

        return self.env.reset(seed=seed)
