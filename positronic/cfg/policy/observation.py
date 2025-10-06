import configuronic as cfn


@cfn.config()
def end_effector(resolution: tuple[int, int]):
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['grip'],
        left=('left.image', resolution),
        right=('right.image', resolution),
    )


end_effector_224 = end_effector.override(resolution=(224, 224))
end_effector_384 = end_effector.override(resolution=(384, 384))
end_effector_352x192 = end_effector.override(resolution=(352, 192))


# State for back and front camera used mostly in simulation
@cfn.config()
def end_effector_back_front():
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['grip'],
        back=('image.back', (352, 192)),
        front=('image.front', (352, 192)),
    )


# Similiar to end_effector but with additional frames listed for compatibility
@cfn.config()
def end_effector_mem15():
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['grip'],
        left=('left.image', (352, 192)),
        right=('right.image', (352, 192)),
        left_15=('left.image', (352, 192)),
        right_15=('right.image', (352, 192)),
    )


@cfn.config(state=['robot_state.ee_pose', 'grip'])
def franka_mujoco_stackcubes(state):
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=state,
        left=('image.handcam_left', (224, 224)),
        side=('image.back_view', (224, 224)),
    )


@cfn.config()
def pi0():
    from positronic.policy.observation import ObservationEncoder

    return ObservationEncoder(
        state_features=['robot_state.ee_pose', 'grip'],
        image=('image.left', (224, 224)),
        side=('image.side', (224, 224)),
    )
