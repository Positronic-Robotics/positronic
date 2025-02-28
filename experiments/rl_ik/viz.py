from typing import Callable

import numpy as np
import cv2


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

    # Store original dimensions to restore later
    original_width = env.unwrapped.img_width
    original_height = env.unwrapped.img_height

    # Set dimensions for rendering
    env.unwrapped.img_width = width
    env.unwrapped.img_height = height
    env.unwrapped.center_x = width // 2
    env.unwrapped.center_y = height // 2
    env.unwrapped.scale_factor = min(width, height) / 6  # Scale to fit in view

    # Reset environment
    obs, _ = env.reset()

    done = False
    while not done:
        # Render the current state using env.render()
        frame = env.render()

        # OpenCV uses BGR format, convert to RGB for consistency
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        # Step environment
        action = policy(obs)
        obs, _, done, _, _ = env.step(action)

    # Restore original dimensions
    env.unwrapped.img_width = original_width
    env.unwrapped.img_height = original_height
    env.unwrapped.center_x = original_width // 2
    env.unwrapped.center_y = original_height // 2
    env.unwrapped.scale_factor = min(original_width, original_height) / 6

    return np.array(frames)
