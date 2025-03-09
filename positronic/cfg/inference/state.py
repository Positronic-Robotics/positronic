import ironic as ir

from positronic.inference.state import StateEncoder, ImageEncodingConfig, StateEncodingConfig

end_effector= ir.Config(
    StateEncoder,
    state_output_key='observation.state',
    images=[
        ImageEncodingConfig(
            key='first.image',
            output_key='observation.images.front',
            resize=[352, 192]
        ),
        ImageEncodingConfig(
            key='second.image',
            output_key='observation.images.back',
            resize=[352, 192]
        ),
    ],
    state=[
        'grip',
    ]
)
