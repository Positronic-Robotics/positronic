from typing import Callable

import matplotlib
import numpy as np

matplotlib.use('agg')  # Use agg backend for offscreen rendering

import matplotlib.pyplot as plt  # noqa: E402


def render_frames(env,
                  policy: Callable[[np.ndarray], np.ndarray],
                  width: int = 640,
                  height: int = 480,
                  show_trajectory: bool = True) -> np.ndarray:
    """Render frames of the robotic arm following a trajectory.

    Args:
        env: RoboticArmEnv instance
        policy: Function that takes observation and returns action
        width: Width of output frames
        height: Height of output frames
        show_trajectory: Whether to show the trajectory

    Returns:
        Numpy array of shape (num_frames, height, width, 3) containing rendered frames in RGB format
    """
    frames = []

    # Create figure with white background
    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111)

    # Reset environment
    obs, _ = env.reset()

    # Set fixed plot limits to prevent auto-scaling
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    done = False
    while not done:
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_facecolor('white')

        # Draw trajectory if enabled
        if show_trajectory:
            target_pos = env.unwrapped.target_pos
            if len(env.unwrapped.trajectory) > 0:
                trajectory_points = [target_pos]
                trajectory_points.extend(env.unwrapped.trajectory)
                trajectory_points = np.array(trajectory_points)
                ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'gray', alpha=0.5, linewidth=2)

        # Draw robot arm
        # Base joint
        base_pos = np.array([0., 0.])

        # First link
        joint_pos = env.unwrapped.L[0] * np.array(
            [np.cos(env.unwrapped.state.thetas[0]),
             np.sin(env.unwrapped.state.thetas[0])])

        # Second link
        ee_pos = joint_pos + env.unwrapped.L[1] * np.array([
            np.cos(env.unwrapped.state.thetas[0] + env.unwrapped.state.thetas[1]),
            np.sin(env.unwrapped.state.thetas[0] + env.unwrapped.state.thetas[1])
        ])

        # Draw links
        ax.plot([base_pos[0], joint_pos[0]], [base_pos[1], joint_pos[1]], 'b-', linewidth=3, label='Link 1')
        ax.plot([joint_pos[0], ee_pos[0]], [joint_pos[1], ee_pos[1]], 'b-', linewidth=3, label='Link 2')

        # Draw joints
        ax.plot(base_pos[0], base_pos[1], 'ko', markersize=8)
        ax.plot(joint_pos[0], joint_pos[1], 'ko', markersize=8)
        ax.plot(ee_pos[0], ee_pos[1], 'ko', markersize=8)

        # Draw target
        ax.plot(env.unwrapped.target_pos[0], env.unwrapped.target_pos[1], 'ro', markersize=8)

        # Remove axes for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw the canvas and get the RGB data directly
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

        frames.append(data)

        # Step environment
        action = policy(obs)
        obs, _, done, _, _ = env.step(action)

    plt.close(fig)
    return np.array(frames)
