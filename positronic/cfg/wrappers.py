import configuronic as cfn

from positronic.policy.wrappers import ChunkedSchedule, ErrorRecovery, TemporalFrameStack

chunked_schedule = cfn.Config(ChunkedSchedule)
error_recovery = cfn.Config(ErrorRecovery)


@cfn.config()
def temporal_frame_stack(image_keys: tuple[str, ...], offsets_sec: tuple[float, ...]):
    return TemporalFrameStack(image_keys=tuple(image_keys), offsets_sec=tuple(offsets_sec))
