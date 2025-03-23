import ironic as ir

from positronic.inference.state import StateEncoder, ImageEncodingConfig, StateEncodingConfig

end_effector= ir.Config(
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
    ],
    state=[
        'grip',  # fake grip
    ]
)


end_effector_mem15 = ir.Config(
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