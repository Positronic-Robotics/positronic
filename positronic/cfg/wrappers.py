import configuronic as cfn

from positronic.policy.wrappers import ChunkedSchedule, ErrorRecovery, TemporalFrameStack


@cfn.config()
def chunked_schedule():
    return ChunkedSchedule()


@cfn.config()
def error_recovery():
    return ErrorRecovery()


@cfn.config()
def temporal_frame_stack(image_keys: tuple[str, ...], offsets_sec: tuple[float, ...]):
    return TemporalFrameStack(image_keys=tuple(image_keys), offsets_sec=tuple(offsets_sec))
