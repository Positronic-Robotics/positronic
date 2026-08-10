"""Reduce the wall-clock telemetry sidecars written by ``positronic eval run --timing`` into a pass report.

Everything is read back from the per-process sidecar files under ``<run_dir>/telemetry/`` — the harness
process's nested spans (the pass, each episode, and its reset / env-step / materialize / record-IO / inference
phases) and its machine-load stats stream, plus the env server's own spans file (its step decomposed into
physics / render). A rollout cannot recover these wall costs from its own timestamps, so the report is an
offline reduce over the raw spans and samples: the per-phase wall split, the env-step split, the
inference-latency distribution, the rollout duration over span wall, and the sim box's GPU load. That last
ratio is read as a real-time factor only where the pass ran on a virtual clock; an attended pass is paced by
its operator on the wall clock, so the same ratio is the share of window wall spent in rollouts. A directory
holding passes of both kinds is refused rather than reduced into one.
The policy endpoint (a different box) folds in from an optional ``nvidia-smi dmon`` log.
"""

import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

import configuronic as cfn
import numpy as np
import pos3

from positronic.telemetry import (
    GPU_INDEX,
    GPU_MEM_USED_B,
    GPU_PROC_MEM_B,
    GPU_UTIL_PCT,
    SPANS_SUFFIX,
    STAT_GPU_COUNT,
    STAT_GPUS,
    STAT_T_NS,
    STATS_SUFFIX,
    TELEMETRY_SUBDIR,
    SpanRec,
    read_spans,
    read_stats,
)
from positronic.telemetry_keys import (
    ATTR_EPISODE_ABORTED,
    ATTR_EPISODE_PARTIAL,
    ATTR_EPISODE_VIRTUAL_S,
    ATTR_PASS_FAILED,
    ATTR_PASS_VIRTUAL_CLOCK,
    HARNESS_PROCESS,
    SPAN_ENV_STEP,
    SPAN_EPISODE,
    SPAN_EVAL_PASS,
    SPAN_MATERIALIZE,
    SPAN_POLICY_INFER,
    SPAN_RECORD_IO,
    SPAN_RESET,
)

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024**3

# The ``nvidia-smi dmon`` column names the policy-endpoint log is read through: SM utilisation (%), framebuffer
# use (MiB) and the device index. Named once so header detection and row indexing cannot drift apart.
_DMON_UTIL = 'sm'
_DMON_FB = 'fb'
_DMON_DEVICE = 'gpu'


@dataclass
class GpuSummary:
    """Mean utilisation and peak VRAM for one box.

    ``mean_util_pct`` averages every per-device reading taken, so a device that never answers biases it
    towards the ones that did. ``devices_seen`` and ``box_devices`` carry that bias with the number instead of
    leaving it to be inferred: they are the devices the mean covers and the devices the box holds, and
    ``devices_seen < box_devices`` says outright that the figure describes part of a box. It reads ``None``
    when no device answered at all, which is a box whose GPUs are all unreadable rather than a box with no
    GPUs. A ``dmon`` log declares no complement, so its box is the set of device indices its rows mention —
    a device absent from the log entirely is unknowable there.

    The peaks are box-wide totals — a sample's devices summed, then the max over samples — so each needs a
    sample that saw the whole box, and reads ``None`` when none did. ``peak_proc_vram_gb`` is the share
    attributed to this eval's process tree and needs that attribution on top; it is always ``None`` for a
    ``dmon`` log, which carries none (only the whole box).
    """

    mean_util_pct: float | None
    devices_seen: int
    box_devices: int
    peak_vram_gb: float | None
    peak_proc_vram_gb: float | None


@dataclass
class GpuReport:
    """GPU summaries per box: the sim box (from the recorded stats stream) and the policy endpoint.

    ``sim`` is ``None`` on a CPU sim box (no GPU samples); ``policy`` is ``None`` when no policy ``dmon`` log
    was passed in.
    """

    sim: GpuSummary | None
    policy: GpuSummary | None


class WallWindow(StrEnum):
    """The wall window every share in a report is a fraction of, named by the symbol the report prints.

    ``W_PASS`` is the ``eval.pass`` span, which covers the whole pass including the setup and teardown either
    side of its episodes. ``W_EPISODES`` is the fallback for a run whose pass span never closed: the wall from
    its first episode's start to its last one's end. That window is narrower — whatever ran before the first
    episode or after the last is outside it — so a share of it is not a share of a pass, and the two carry
    different names for that reason.
    """

    W_PASS = 'W_pass'
    W_EPISODES = 'W_episodes'


@dataclass
class WallSplit:
    """Each phase's share of the report's wall window.

    ``overhead`` is the within-episode wall unattributed to a measured phase (including the recorder's
    parquet/video close flush, which runs after record IO); ``between_episodes`` is the inter-episode wall
    (session teardown, homing, world rebuild) inside the window between one episode's finish and the next's
    start. The measured phases plus these two cover the window minus any aborted-episode wall, which is
    excluded from it entirely.
    """

    reset: float
    env_step: float
    policy_wait: float
    record_io: float
    overhead: float
    between_episodes: float


@dataclass
class EnvStepSplit:
    """Fractions of the client-observed env-step wall, from the env server's own decomposition.

    ``phases`` maps each env-reported phase (physics, render, and the ``server_other`` residual — the env owns
    the phase set, so it is a map, not fixed fields) to its share of the client step wall; ``wire`` is the
    client step wall minus the server's whole in-step wall and the client materialisation (socket + codec);
    ``materialize`` is the client-side observation materialisation. ``None`` when no env server reported a
    decomposition (a native sim carries none).
    """

    phases: dict[str, float]
    wire: float
    materialize: float


@dataclass
class PassReport:
    """Pass-level wall-clock roll-up reduced from the recorded telemetry spans and stats.

    ``window`` names which wall ``wall_s`` measures and which every share in ``wall_split``, and the real-time
    factor, are fractions of. It rides with the numbers because a run killed mid-pass reduces against the
    narrower episode window, and a figure normalised by that one is not comparable with a pass figure.
    """

    episodes: int
    window: WallWindow
    wall_s: float
    # ``real_time_factor`` is sim-s per wall-s and exists only under a virtual clock.
    # An attended pass runs on the wall clock, where the ratio is the share of window wall spent in rollouts.
    virtual_clock: bool
    real_time_factor: float | None
    rollout_wall_share: float
    infer_calls: int
    infer_p50_ms: float
    infer_p95_ms: float
    wall_split: WallSplit
    env_step_split: EnvStepSplit | None
    gpu: GpuReport


def _dur_s(span: SpanRec) -> float:
    return (span.end_ns - span.start_ns) / 1e9


def _read_spans_dir(telemetry_dir: Path) -> list[SpanRec]:
    """Every span across all per-process ``*.spans.jsonl`` files under the telemetry dir."""
    spans: list[SpanRec] = []
    for path in sorted(telemetry_dir.glob(f'*{SPANS_SUFFIX}')):
        spans.extend(read_spans(path))
    return spans


def _read_stats_dir(telemetry_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for path in sorted(telemetry_dir.glob(f'*{STATS_SUFFIX}')):
        samples.extend(read_stats(path))
    return samples


def _gpu_summary_from_stats(stats: list[dict], windows: list[tuple[int, int]]) -> GpuSummary | None:
    """The sim box's GPU summary from the machine-load stream. Only samples taken inside one of the report's
    wall windows count — a reused directory carries an earlier (possibly killed) run's samples, the stats twin
    of the orphan-episode exclusion. ``None`` when no counted sample held a device at all (a CPU sim box); a
    box whose devices all refused their queries is a summary with nothing measured, not a CPU box."""
    in_window = [s for s in stats if any(start <= int(s[STAT_T_NS]) <= end for start, end in windows)]
    # The recorded count is the box's NVML handle count, so it stands whether or not any device answered:
    # reading it across every sample, including the empty ones, separates an unreadable GPU box from a CPU one.
    box_devices = max((int(s[STAT_GPU_COUNT]) for s in in_window), default=0)
    utils: list[float] = []
    mem: list[float] = []
    proc: list[float] = []
    seen: set[int] = set()
    for sample in in_window:
        # A directory resumed on a box with fewer GPUs holds samples of another machine. This summary reports
        # one complement, so those readings belong to no figure in it — not the mean either, which would
        # otherwise average two boxes under one box's name.
        if int(sample[STAT_GPU_COUNT]) != box_devices:
            continue
        gpus = sample.get(STAT_GPUS, [])
        if not gpus:
            continue
        utils.extend(float(gpu[GPU_UTIL_PCT]) for gpu in gpus)
        seen.update(int(gpu[GPU_INDEX]) for gpu in gpus)
        # Either peak is a box-wide total over that complement, so a sample counts only when it carries all of
        # it: a device whose NVML query errors mid-run is dropped from the sample entirely rather than left
        # ``None``, and its total would cover part of the box and be read as the whole of it. The mean is
        # per-device, so those readings still count. The process peak needs per-device attribution on top, so
        # a ``None`` reading drops the sample from that peak alone.
        if len(gpus) != box_devices:
            continue
        mem.append(sum(float(gpu[GPU_MEM_USED_B]) for gpu in gpus))
        if all(gpu.get(GPU_PROC_MEM_B) is not None for gpu in gpus):
            proc.append(sum(float(gpu[GPU_PROC_MEM_B]) for gpu in gpus))
    if box_devices == 0:
        return None
    if not utils:
        logger.warning(
            "none of the box's %d GPU(s) answered a single machine-load query (all refused, as an unsupported "
            'MIG device does); the box is reported with no metrics rather than as CPU-only',
            box_devices,
        )
    elif not mem:
        logger.warning(
            "no machine-load sample carried all %d of the box's GPUs (an NVML query refused mid-run?); both "
            'peaks are unavailable, and mean utilisation covers the %d device(s) that answered',
            box_devices,
            len(seen),
        )
    return GpuSummary(
        mean_util_pct=float(np.mean(utils)) if utils else None,
        devices_seen=len(seen),
        box_devices=box_devices,
        peak_vram_gb=(max(mem) / _BYTES_PER_GB) if mem else None,
        peak_proc_vram_gb=(max(proc) / _BYTES_PER_GB) if proc else None,
    )


def _dmon_column_indices(names: list[str], log_path: Path) -> tuple[int, int, int | None]:
    """Positions of the ``sm``, ``fb`` and (where present) ``gpu`` columns in a dmon name header.

    A header carrying ``sm`` but no ``fb`` (plain ``dmon`` / ``-s u``) fails loudly: every row would be
    skipped and the policy box would report no peak VRAM at all, as if none had been asked for.
    """
    if _DMON_FB not in names:
        raise ValueError(
            f'{log_path}: nvidia-smi dmon log has an `{_DMON_UTIL}` column but no `{_DMON_FB}` (framebuffer) '
            f'column, so peak VRAM cannot be read and every sample would be dropped. Re-collect it with '
            f'`nvidia-smi dmon -s um` (u=utilisation, m=memory), which emits the `{_DMON_FB}` column.'
        )
    device_idx = names.index(_DMON_DEVICE) if _DMON_DEVICE in names else None
    return names.index(_DMON_UTIL), names.index(_DMON_FB), device_idx


def _dmon_cell(row: list[str], index: int) -> float | None:
    """One dmon cell as a float, or ``None`` where the column is missing or not a number — a device that
    cannot report a metric prints ``-`` for it alone."""
    try:
        return float(row[index])
    except (IndexError, ValueError):
        return None


def _dmon_readings(log_path: Path) -> Iterator[tuple[str, float | None, float | None]]:
    """Each data row of a ``dmon`` log as ``(device, sm, fb)``, in file order.

    Each metric is read on its own, so an unreadable column costs only itself: a row carrying utilisation
    but no framebuffer still contributes its utilisation. A row whose metrics are both unreadable still
    yields its device, because the device index is what marks a sampling interval; dropping it whole would
    merge two intervals. A row before the name header, or one too short to carry a device index, yields
    nothing.
    """
    sm_idx: int | None = None
    fb_idx: int | None = None
    gpu_idx: int | None = None
    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            # The name header carries the column names ('sm', 'fb', ...); the units line ('%', 'MB') has no 'sm'.
            names = stripped.lstrip('#').split()
            if _DMON_UTIL in names:
                sm_idx, fb_idx, gpu_idx = _dmon_column_indices(names, log_path)
            continue
        if sm_idx is None or fb_idx is None:
            continue
        row = line.split()
        try:
            device = row[gpu_idx] if gpu_idx is not None else '0'
        except IndexError:
            continue
        yield device, _dmon_cell(row, sm_idx), _dmon_cell(row, fb_idx)


def _parse_dmon(log_path: Path) -> GpuSummary:
    """Mean SM utilisation and peak framebuffer (GB) from a policy endpoint's ``nvidia-smi dmon`` log.

    Column layout varies — ``-o DT`` prepends date/time, ``-s u`` adds encoder/decoder (and, on newer
    drivers, JPEG/OFA) columns before the framebuffer — so the positions are read from the ``# ... sm ...
    fb ...`` name header rather than hard-coded. ``sm`` is SM utilisation (%) and ``fb`` the framebuffer
    use (MiB). A log whose header has ``sm`` but no ``fb`` (plain ``dmon`` / ``-s u``) fails loudly rather
    than dropping every row. A dmon log carries no per-process attribution, so ``peak_proc_vram_gb`` is
    ``None``.
    """
    utils: list[float] = []
    # dmon prints one row per device per sampling interval; peak VRAM is the box-wide total at one instant,
    # so rows group into cycles on the ``gpu`` index (a repeating index opens the next cycle) and the peak is
    # the max over per-cycle sums — matching the sim summary's per-sample device sum.
    cycle_fb: dict[str, float] = {}
    cycles: list[dict[str, float]] = []
    # Every device index the log mentions is the box's complement; the ones that reported utilisation are
    # what the mean covers. A device that never reports it counts towards the first and not the second — and
    # the two metrics are tracked apart, so a device reporting one and not the other still contributes it.
    devices: set[str] = set()
    answered: set[str] = set()
    # Every device the open cycle has covered, readable metrics or not: an unreadable row marks the interval
    # boundary just as well as a readable one, and only the readable ones carry a total.
    cycle_devices: set[str] = set()

    def flush_cycle() -> None:
        if cycle_fb:
            cycles.append(dict(cycle_fb))
            cycle_fb.clear()
        cycle_devices.clear()

    for device, sm, fb in _dmon_readings(log_path):
        if device in cycle_devices:
            flush_cycle()
        cycle_devices.add(device)
        devices.add(device)
        if sm is not None:
            utils.append(sm)
            answered.add(device)
        if fb is not None:
            cycle_fb[device] = fb
    flush_cycle()
    # The complement is known only once the whole log is read, so completeness is decided here rather than at
    # each flush. A cycle missing a device is a partial box, and the max over partial sums is not a peak: one
    # device's 12 GiB with its neighbour unread outranks a complete 11 GiB instant and reports as the box.
    complete = [sum(cycle.values()) for cycle in cycles if len(cycle) == len(devices)]
    if cycles and not complete:
        logger.warning(
            '%s: no dmon sampling interval carried all %d device(s) the log mentions; peak VRAM is unavailable',
            log_path,
            len(devices),
        )
    return GpuSummary(
        mean_util_pct=float(np.mean(utils)) if utils else None,
        devices_seen=len(answered),
        box_devices=len(devices),
        peak_vram_gb=(max(complete) / 1024.0) if complete else None,
        peak_proc_vram_gb=None,
    )


@dataclass
class _EpisodeTiming:
    """One episode's wall aggregate reduced from its span subtree; fields are seconds unless named otherwise.
    ``env_step_s`` is the client env-step wall (materialisation included); ``materialize_s`` is that
    materialisation alone; ``overhead_s`` is the episode wall left unattributed after the measured phases."""

    wall_s: float
    virtual_s: float
    reset_s: float
    env_step_s: float
    materialize_s: float
    record_io_s: float
    policy_wait_s: float
    overhead_s: float
    infer_ms: list[float]


def _episode_timing(episode: SpanRec, children: dict[str, list[SpanRec]]) -> _EpisodeTiming:
    kids = children.get(episode.span_id, [])
    reset_s = sum(_dur_s(k) for k in kids if k.name == SPAN_RESET)
    env_step_s = sum(_dur_s(k) for k in kids if k.name == SPAN_ENV_STEP)
    record_io_s = sum(_dur_s(k) for k in kids if k.name == SPAN_RECORD_IO)
    infer_ms = [_dur_s(k) * 1000.0 for k in kids if k.name == SPAN_POLICY_INFER]
    materialize_s = sum(
        _dur_s(child)
        for k in kids
        if k.name == SPAN_ENV_STEP
        for child in children.get(k.span_id, [])
        if child.name == SPAN_MATERIALIZE
    )
    policy_wait_s = sum(infer_ms) / 1000.0
    wall_s = _dur_s(episode)
    measured = reset_s + env_step_s + record_io_s + policy_wait_s
    return _EpisodeTiming(
        wall_s=wall_s,
        virtual_s=float(episode.attrs.get(ATTR_EPISODE_VIRTUAL_S, 0.0)),
        reset_s=reset_s,
        env_step_s=env_step_s,
        materialize_s=materialize_s,
        record_io_s=record_io_s,
        policy_wait_s=policy_wait_s,
        overhead_s=max(wall_s - measured, 0.0),
        infer_ms=infer_ms,
    )


def _env_step_split(spans: list[SpanRec], episodes: list[SpanRec], env_step_sum: float, materialize_sum: float):
    """The env-step decomposition from the env server's own spans: a server ``env.step`` (recorded in the env
    process's own file, so ``process`` is not the harness) with physics/render children. Only server steps that
    start inside a completed episode's wall window count — an aborted rollout's steps would otherwise skew
    fractions whose denominator covers completed episodes only. ``None`` when no such span exists (a native sim
    reports no decomposition)."""

    def in_completed_episode(ts_ns: int) -> bool:
        return any(e.start_ns <= ts_ns <= e.end_ns for e in episodes)

    server_steps = [
        s
        for s in spans
        if s.name == SPAN_ENV_STEP and s.process != HARNESS_PROCESS and in_completed_episode(s.start_ns)
    ]
    server_step_sum = sum(_dur_s(s) for s in server_steps)
    if not server_step_sum or not env_step_sum:
        return None
    server_ids = {s.span_id for s in server_steps}
    phase_sums: dict[str, float] = defaultdict(float)
    for span in spans:
        if span.parent_id in server_ids:
            phase_sums[span.name] += _dur_s(span)
    phases = {name: total / env_step_sum for name, total in phase_sums.items()}
    server_other = max(server_step_sum - sum(phase_sums.values()), 0.0)
    if server_other:
        phases['server_other'] = server_other / env_step_sum
    return EnvStepSplit(
        phases=phases,
        wire=(env_step_sum - server_step_sum - materialize_sum) / env_step_sum,
        materialize=materialize_sum / env_step_sum,
    )


def _episode_windows(episodes: list[SpanRec]) -> dict[str | None, tuple[int, int]]:
    """One wall window per run whose ``eval.pass`` span never closed, keyed by the pass span the episodes name
    as their parent — from that run's first episode start to its last episode end.

    Grouping by parent is what keeps two killed runs appended to one directory apart: each contributes its own
    window, so the dead wall between them falls outside both, exactly as the gap between two pass spans does.
    Aborted episodes count towards the window, because they are subtracted from it by the same rule that
    subtracts them from a pass span's wall.
    """
    by_parent: dict[str | None, list[SpanRec]] = defaultdict(list)
    for episode in episodes:
        by_parent[episode.parent_id].append(episode)
    return {
        parent: (min(e.start_ns for e in group), max(e.end_ns for e in group)) for parent, group in by_parent.items()
    }


def _passes_virtual_clock(passes: list[SpanRec]) -> bool:
    """Which clock the passes were measured on, refusing a directory that holds both.

    Absent on a sidecar written before the pass carried its clock; every one of those is a sim sweep.
    """
    clocks = {bool(p.attrs.get(ATTR_PASS_VIRTUAL_CLOCK, True)) for p in passes}
    if len(clocks) > 1:
        raise ValueError(
            'Telemetry holds passes measured on different clocks: sim seconds and wall seconds cannot be '
            'summed into one ratio, so no label on the result would be true. Reduce each run separately.'
        )
    return clocks.pop() if clocks else True


def _build_report(spans: list[SpanRec], stats: list[dict], policy_gpu: GpuSummary | None) -> PassReport:
    children: dict[str, list[SpanRec]] = defaultdict(list)
    for span in spans:
        if span.parent_id is not None:
            children[span.parent_id].append(span)

    # The denominator is the pass span's wall (summed if several passes appended to one dir), so inter-episode
    # teardown counts in it; the wall gap between two separate passes falls outside every pass span. A run
    # killed or preempted mid-pass never writes that span, so it falls back to the wall its complete episodes
    # span — narrower, and named W_episodes rather than W_pass because a share of it is not a share of a pass.
    all_episodes = [s for s in spans if s.name == SPAN_EPISODE]
    passes = [p for p in spans if p.name == SPAN_EVAL_PASS]
    if passes:
        window = WallWindow.W_PASS
        windows = {p.span_id: (p.start_ns, p.end_ns) for p in passes}
    else:
        window = WallWindow.W_EPISODES
        windows = _episode_windows(all_episodes)
        if windows:
            logger.warning(
                'no %s span closed (a killed or preempted run?); reducing over the wall its complete episodes '
                'span, so every share is of %s rather than of a pass window',
                SPAN_EVAL_PASS,
                window.value,
            )
    if not windows:
        raise ValueError(
            f'the recorded telemetry holds {len(spans)} span(s) but no closed `{SPAN_EVAL_PASS}` span and no '
            f'`{SPAN_EPISODE}` span, so there is nothing to reduce (killed before its first episode finished?)'
        )
    # An aborted episode's wall is subtracted out: the episode is dropped as invalid data (its virtual time,
    # phases and infers never reduce), so its wall must also leave every normalised figure rather than land in
    # ``between_episodes`` and deflate the policy-wait / real-time factors.
    aborted_wall = float(
        sum(_dur_s(s) for s in all_episodes if s.attrs.get(ATTR_EPISODE_ABORTED, False) and s.parent_id in windows)
    )
    wall = float(sum(end - start for start, end in windows.values())) / 1e9 - aborted_wall
    # A failed pass still reduces — its partial window, episodes and samples are real recorded data — but
    # never silently: the mix is named so a skewed-looking split has its explanation on the console.
    failed = sum(bool(p.attrs.get(ATTR_PASS_FAILED, False)) for p in passes)
    if failed:
        logger.warning(
            '%d of %d pass(es) failed mid-run; their partial windows are included in the report', failed, len(passes)
        )
    # Only episodes belonging to a window reduce: where one pass completed, a killed earlier run's episodes are
    # in the same reused directory but under no window of their own, and would inflate every normalised figure.
    episodes = [s for s in all_episodes if not s.attrs.get(ATTR_EPISODE_ABORTED, False)]
    # A partial episode (its rollout failed mid-run) is kept, not dropped: its finished phases are real wall
    # and must attribute rather than fall into ``between_episodes``. Named, like ``pass.failed``, so the split
    # is read with its incomplete window in view.
    partial = sum(bool(e.attrs.get(ATTR_EPISODE_PARTIAL, False)) for e in episodes)
    if partial:
        logger.warning(
            '%d episode(s) did not run to completion (a failed pass?); their finished phases are included', partial
        )
    orphans = sum(e.parent_id not in windows for e in episodes)
    if orphans:
        logger.warning('%d episode(s) belong to no completed pass (a killed run?); excluded from the report', orphans)
        episodes = [e for e in episodes if e.parent_id in windows]
    timings = [_episode_timing(e, children) for e in episodes]

    episode_wall_sum = float(sum(t.wall_s for t in timings))
    env_step_sum = float(sum(t.env_step_s for t in timings))
    materialize_sum = float(sum(t.materialize_s for t in timings))
    all_infer_ms = np.array([ms for t in timings for ms in t.infer_ms], dtype=float)

    def phase_fraction(total: float) -> float:
        return (total / wall) if wall else 0.0

    wall_split = WallSplit(
        reset=phase_fraction(sum(t.reset_s for t in timings)),
        env_step=phase_fraction(env_step_sum),
        policy_wait=phase_fraction(sum(t.policy_wait_s for t in timings)),
        record_io=phase_fraction(sum(t.record_io_s for t in timings)),
        overhead=phase_fraction(sum(t.overhead_s for t in timings)),
        between_episodes=phase_fraction(wall - episode_wall_sum),
    )
    virtual_clock = _passes_virtual_clock(passes)
    rollout_share = (sum(t.virtual_s for t in timings) / wall) if wall else 0.0
    return PassReport(
        episodes=len(episodes),
        window=window,
        wall_s=wall,
        virtual_clock=virtual_clock,
        real_time_factor=rollout_share if virtual_clock else None,
        rollout_wall_share=rollout_share,
        infer_calls=int(all_infer_ms.size),
        infer_p50_ms=float(np.percentile(all_infer_ms, 50)) if all_infer_ms.size else 0.0,
        infer_p95_ms=float(np.percentile(all_infer_ms, 95)) if all_infer_ms.size else 0.0,
        wall_split=wall_split,
        env_step_split=_env_step_split(spans, episodes, env_step_sum, materialize_sum),
        gpu=GpuReport(sim=_gpu_summary_from_stats(stats, list(windows.values())), policy=policy_gpu),
    )


def _share_row(name: str, fraction: float) -> str:
    """One indented ``name    12.3%`` row of a split.

    The label column fits the longest name any split renders (``between_episodes``), so the percentages line
    up in one column across the wall and env-step splits.
    """
    return f'  {name:<16} {fraction * 100:>6.1f}%'


def _render_ratio(report: PassReport) -> str:
    """The rollout-over-pass ratio, named for the clock it was measured on."""
    if report.virtual_clock:
        assert report.real_time_factor is not None  # populated together with the flag
        return f'real-time factor:    {report.real_time_factor * 100:>6.1f}% (sim-s per wall-s)'
    return f'rollout wall share:  {report.rollout_wall_share * 100:>6.1f}% (wall-s in rollouts per wall-s)'


def _render(report: PassReport) -> str:
    window = report.window.value
    lines = []
    # The console warning is lost the moment this text is pasted anywhere, and the numbers under a narrower
    # window read exactly like pass numbers, so the report says which window it reduced over in its own body.
    if report.window is WallWindow.W_EPISODES:
        lines.append(
            f'no {SPAN_EVAL_PASS} span closed (a killed or preempted run?): {window} is the wall from the first '
            f'complete episode to the last, and every share below is of that, not of a pass'
        )
    lines += [
        f'episodes:            {report.episodes}',
        f'{window + " (wall):":<21}{report.wall_s:.1f} s ({report.wall_s / 3600:.2f} h)',
        _render_ratio(report),
        f'infer calls:         {report.infer_calls}',
        f'infer p50 / p95:     {report.infer_p50_ms:.1f} / {report.infer_p95_ms:.1f} ms',
        f'wall split (share of {window}):',
    ]
    for f in fields(WallSplit):
        share = getattr(report.wall_split, f.name)
        row = _share_row(f.name, share)
        # The wall share blocked on inference is also this eval's serving-capacity figure: how many such evals
        # one policy server could keep fed, if inference is what saturates it.
        if f.name == 'policy_wait' and share:
            row += f'  -> ~{1 / share:.1f} sims per policy server'
        lines.append(row)
    if report.env_step_split is not None:
        split = report.env_step_split
        lines.append('env-step split (share of env_step):')
        lines += [_share_row(name, frac) for name, frac in split.phases.items()]
        lines.append(_share_row('wire', split.wire))
        lines.append(_share_row('materialize', split.materialize))
    for f in fields(GpuReport):
        summary = getattr(report.gpu, f.name)
        if summary is not None:
            util = f'util {summary.mean_util_pct:.0f}%' if summary.mean_util_pct is not None else 'util unavailable'
            # A mean over part of a box says so where it is read, not only in the log: the coverage rides with
            # the number, and appears exactly when it qualifies it.
            if summary.devices_seen < summary.box_devices:
                util += f' over {summary.devices_seen} of {summary.box_devices} GPUs'
            peak = f'{summary.peak_vram_gb:.1f} GB' if summary.peak_vram_gb is not None else 'unavailable'
            line = f'gpu[{f.name}]: {util}  peak VRAM {peak}'
            if summary.peak_proc_vram_gb is not None:
                line += f'  (this eval {summary.peak_proc_vram_gb:.1f} GB)'
            lines.append(line)
    return '\n'.join(lines)


@cfn.config(gpu_policy_log=None)
def timing_report(run_dir: str, gpu_policy_log: str | None):
    """Reduce the telemetry sidecars under ``<run_dir>/telemetry/`` into a pass report.

    ``run_dir`` may be an ``s3://`` URI — the documented Nebius eval path writes there — in which case it is
    pulled local first, mirroring how the eval command syncs its output. The sim box's GPU load comes from the
    recorded machine-load stats stream; ``gpu_policy_log`` is an optional ``nvidia-smi dmon -s um`` log for the
    policy endpoint (a different box) folded in the same way. Writes ``timing_summary.json`` under the input —
    for an ``s3://`` input to the sibling key ``<run_dir>.timing_summary.json`` (pos3 forbids uploading inside
    the downloaded prefix) — and prints the report.
    """
    root = Path(pos3.download(run_dir)) if '://' in run_dir else Path(run_dir)
    telemetry_dir = root / TELEMETRY_SUBDIR
    spans = _read_spans_dir(telemetry_dir) if telemetry_dir.is_dir() else []
    if not spans:
        raise ValueError(f'no telemetry under {telemetry_dir} (recorded without --timing?)')

    policy_gpu = _parse_dmon(Path(gpu_policy_log)) if gpu_policy_log is not None else None
    report = _build_report(spans, _read_stats_dir(telemetry_dir), policy_gpu)
    summary_path = root / 'timing_summary.json'
    summary_path.write_text(json.dumps(asdict(report), indent=2))
    if '://' in run_dir:
        # A remote input was only downloaded, so the write above lands in the local cache. pos3 rejects an
        # upload inside a registered download prefix, so push the summary to a sibling key next to the run dir.
        pos3.upload(f'{run_dir.rstrip("/")}.timing_summary.json', summary_path, delete=False)
    logger.info(f'wrote {summary_path}')
    print(_render(report))
