import configuronic as cfn

from positronic.policy.wrappers import ChunkedSchedule, ErrorRecovery, TemporalFrameStack

chunked_schedule = cfn.Config(ChunkedSchedule)
error_recovery = cfn.Config(ErrorRecovery)
temporal_frame_stack = cfn.Config(TemporalFrameStack)
