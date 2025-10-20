import configuronic as cfn


@cfn.config()
def end_effector(resolution: tuple[int, int]):
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['grip'], images={'left': ('left.image', resolution), 'right': ('right.image', resolution)}
    )


end_effector_224 = end_effector.override(resolution=(224, 224))
end_effector_384 = end_effector.override(resolution=(384, 384))
end_effector_352x192 = end_effector.override(resolution=(352, 192))


# State for back and front camera used mostly in simulation
@cfn.config()
def end_effector_back_front():
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['grip'], images={'back': ('image.back', (352, 192)), 'front': ('image.front', (352, 192))}
    )


# Similiar to end_effector but with additional frames listed for compatibility
@cfn.config()
def end_effector_mem15():
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['grip'],
        images={
            'left': ('left.image', (352, 192)),
            'right': ('right.image', (352, 192)),
            'left_15': ('left.image', (352, 192)),
            'right_15': ('right.image', (352, 192)),
        },
    )


@cfn.config(state=['robot_state.ee_pose', 'grip'])
def franka_mujoco_stackcubes(state):
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=state,
        images={'left': ('image.handcam_left', (224, 224)), 'side': ('image.back_view', (224, 224))},
    )


@cfn.config()
def pi0():
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['robot_state.ee_pose', 'grip'],
        images={'left': ('image.left', (224, 224)), 'side': ('image.side', (224, 224))},
    )


@cfn.config()
def openpi_sim(image_size=(224, 224)):
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['robot_state.q', 'grip'],
        images={'exterior': ('image.back_view', image_size), 'wrist': ('image.handcam_left', image_size)},
    )


@cfn.config(exterior_camera='image.exterior', wrist_camera='image.wrist', image_size=(224, 224))
def openpi_droid(exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    """DROID observation encoder using joint positions."""
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['robot_state.q', 'grip'],
        images={'exterior': (exterior_camera, image_size), 'wrist': (wrist_camera, image_size)},
    )
