import configuronic as cfn
from pimm.inference.state import ImageEncodingConfig, StateEncoder


@cfn.config
def end_effector(resolution: tuple[int, int]):
    return StateEncoder(
        state_output_key='observation.state',
        images=[
            ImageEncodingConfig(
                key='left.image',
                output_key='observation.images.left',
                resize=resolution
            ),
            ImageEncodingConfig(
                key='right.image',
                output_key='observation.images.right',
                resize=resolution
            ),
        ],
        state=[
            'grip',  # fake grip
        ]
    )


end_effector_224 = end_effector.override(resolution=(224, 224))
end_effector_384 = end_effector.override(resolution=(384, 384))
end_effector_352x192 = end_effector.override(resolution=(352, 192))


# State for back and front camera used mostly in simulation
end_effector_back_front = cfn.Config(
    StateEncoder,
    state_output_key='observation.state',
    images=[
        ImageEncodingConfig(
            key='image.back',
            output_key='observation.images.back',
            resize=[352, 192]
        ),
        ImageEncodingConfig(
            key='image.front',
            output_key='observation.images.front',
            resize=[352, 192]
        ),
    ],
    state=[
        'grip',
    ]
)


# Similiar to end_effector but with additional frames with 15 frame offset
end_effector_mem15 = cfn.Config(
    StateEncoder,
    state_output_key='observation.state',
    images=[
        ImageEncodingConfig(
            key='left.image',
            output_key='observation.images.left',
            resize=[352, 192]
        ),
        ImageEncodingConfig(
            key='right.image',
            output_key='observation.images.right',
            resize=[352, 192]
        ),
        ImageEncodingConfig(
            key='left.image',
            output_key='observation.images.left_15',
            resize=[352, 192],
            offset=15
        ),
        ImageEncodingConfig(
            key='right.image',
            output_key='observation.images.right_15',
            resize=[352, 192],
            offset=15
        ),
    ],
    state=[
        'grip',  # fake grip
    ]
)
