import configuronic as cfn

from positronic.policy.wrappers import ChunkedSchedule, ErrorRecovery, TemporalFrameStack

chunked_schedule = cfn.Config(ChunkedSchedule)
error_recovery = cfn.Config(ErrorRecovery)
temporal_frame_stack = cfn.Config(TemporalFrameStack)


def _frame_offsets_sec(history_frames: int, stride: int, fps: float) -> tuple[float, ...]:
    """Frame-stack offsets in ascending negative seconds for a strided video-context window.

    ``history_frames`` is the oldest look-back in frames (the magnitude of the most-negative offset).
    Samples the current frame (offset 0) and every ``stride``-th frame back from it, then always pins
    the oldest sample to ``-history_frames``: when ``stride`` divides ``history_frames`` that pin lands
    on the regular grid, otherwise the final gap is shorter than ``stride`` (DreamZero's ``(23, 8)``
    yields frames 23, 16, 8, 0 — the trained -23 oldest, not the stride's -24).

    To reproduce a model trained on contiguous, non-overlapping windows (DreamZero's ACTION_HORIZON /
    RELATIVE_OFFSETS in test_client_AR.py), pass ``history_frames = ACTION_HORIZON - 1`` so the oldest
    frame lands on the window start; the full width would push it one frame back and overlap the prior
    window (out of distribution).
    """
    frames = list(range(0, history_frames, stride)) + [history_frames]
    return tuple(-f / fps for f in reversed(frames))


@cfn.config(image_keys=('image.wrist', 'image.exterior'), fps=15.0)
def video_context_wrappers(history_frames: int, stride: int, image_keys: tuple[str, ...], fps: float):
    """Eval ``wrap`` for video-conditioned policies: error recovery, strided frame-stack context, scheduling.

    The frame stack sits outside the scheduler so it records the named cameras every control tick and
    substitutes a temporal stack sampled per ``history_frames``/``stride``; pair it with a codec whose
    ``horizon`` plays the full returned chunk so the re-query aligns with the frame-stack window.
    """
    frame_stack = TemporalFrameStack(
        image_keys=tuple(image_keys), offsets_sec=_frame_offsets_sec(history_frames, stride, fps)
    )
    return ErrorRecovery() | frame_stack | ChunkedSchedule()


# gyros chunk-causal bridge: 25 frames (24 history + current) at 15 fps spanning the just-executed K=6
# chunk, dense (stride 1) so the bridge encodes them as 6 contiguous subsequent-latents (its context window).
gyros_wrappers = video_context_wrappers.override(history_frames=24, stride=1)
