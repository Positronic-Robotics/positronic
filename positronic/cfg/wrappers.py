import configuronic as cfn

from positronic.policy.wrappers import ChunkedSchedule, ErrorRecovery, TemporalStack

chunked_schedule = cfn.Config(ChunkedSchedule)
error_recovery = cfn.Config(ErrorRecovery)
temporal_stack = cfn.Config(TemporalStack)


@cfn.config()
def default_wrappers():
    """Eval ``wrap`` default: error recovery + chunked scheduling, no temporal stack."""
    return ErrorRecovery() | ChunkedSchedule()


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


@cfn.config(keys=('image.wrist', 'image.exterior', 'robot_state.ee_pose', 'grip'), fps=15.0)
def video_context_wrappers(history_frames: int, stride: int, keys: tuple[str, ...], fps: float):
    """Eval ``wrap`` for video-conditioned policies: error recovery, strided temporal context, scheduling.

    The temporal stack sits outside the scheduler so it records the named ``keys`` every control tick and
    substitutes a stack sampled per ``history_frames``/``stride``; pair it with a codec whose ``horizon``
    plays the full returned chunk so the re-query aligns with the stack window.

    ``keys`` are the entries to stack: the cameras plus any per-frame proprio (e.g.
    ``('robot_state.ee_pose', 'grip')``) so each history step carries its own pose — the trajectory a
    model trained on real per-frame proprio expects. Drop the proprio keys for models whose codec
    consumes only the current proprio.
    """
    stack = TemporalStack(keys=tuple(keys), offsets_sec=_frame_offsets_sec(history_frames, stride, fps))
    return ErrorRecovery() | stack | ChunkedSchedule()
