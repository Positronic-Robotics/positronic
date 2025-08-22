import configuronic as cfn


@cfn.config()
def end_effector(resolution: tuple[int, int], fps: int):
    from positronic.inference.observation import ObservationEncoder, ImageTransform, ToArrayTransform
    return ObservationEncoder(
        transforms=[
            ImageTransform(input_key='left.image', output_key='observation.images.left', resize=resolution),
            ImageTransform(input_key='right.image', output_key='observation.images.right', resize=resolution),
            ToArrayTransform(input_key='grip', n_features=1, output_key='observation.state'),
        ],
        fps=fps
    )


end_effector_224 = end_effector.override(resolution=(224, 224))
end_effector_384 = end_effector.override(resolution=(384, 384))
end_effector_352x192 = end_effector.override(resolution=(352, 192))

@cfn.config(resolution=(224, 224), fps=30)
def end_effector_handcam(resolution: tuple[int, int], fps: int):
    from positronic.inference.observation import ObservationEncoder, ImageTransform, ToArrayTransform
    return ObservationEncoder(
        transforms=[
            ImageTransform(input_key='image.handcam_left', output_key='observation.images.left', resize=resolution),
            ImageTransform(input_key='image.handcam_right', output_key='observation.images.right', resize=resolution),
            ToArrayTransform(input_key='grip', n_features=1, output_key='observation.state'),
        ],
        fps=fps
    )
