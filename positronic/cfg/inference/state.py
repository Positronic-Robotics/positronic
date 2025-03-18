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