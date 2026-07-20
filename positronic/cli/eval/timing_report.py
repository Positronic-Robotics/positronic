"""Reduce a ``timing.jsonl`` (written by ``positronic eval run --timing``) into a pass-level report.

The timer captures only the wall-clock costs a virtual-clock rollout cannot recover; everything else is
read back from the recorded episodes and joined by ``episode_uid``: the virtual duration (for the
real-time factor), the on-disk size (bytes per rollout) and the terminal flag. An optional ``nvidia-smi
dmon`` log per box folds GPU utilisation and peak VRAM into the same report.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import configuronic as cfn
import numpy as np
import pos3

from positronic.dataset.local_dataset import load_all_datasets
from positronic.eval_timing import GPU_LOG_FILENAME, TIMING_FILENAME

logger = logging.getLogger(__name__)


@dataclass
class _RecordedFacts:
    """What the recorded episode contributes to the join, keyed by ``episode_uid``."""

    duration_s: float
    size_mb: float
    success: bool | None  # the env's eval.success verdict; None when it recorded none
    terminated: bool | None  # eval.terminated: the trial ended within budget vs timed out; None if unrecorded
    eval_key: str  # the eval's stable identity (embodiment descriptor) the success oracle is scoped to


@dataclass
class GpuSummary:
    """Utilisation and peak framebuffer use parsed from one ``nvidia-smi dmon -s um`` log."""

    mean_util_pct: float
    peak_vram_gb: float


@dataclass
class PassReport:
    """The pass-level roll-up that replaces the estimated sizing inputs."""

    episodes: int
    wall_pass_s: float
    real_time_factor: float
    policy_busy_fraction: float
    infer_calls: int
    infer_p50_ms: float
    infer_p95_ms: float
    mean_split_fractions: dict[str, float]
    mean_bytes_per_rollout: float
    success_rate: float | None  # None when no episode carried a success verdict (scored downstream)
    gpu: dict[str, GpuSummary]


def _recorded_facts(dataset_dir: Path) -> dict[str, _RecordedFacts]:
    """Join key -> recorded facts for every episode under ``dataset_dir``."""
    dataset = load_all_datasets(dataset_dir)
    facts: dict[str, _RecordedFacts] = {}
    for i in range(len(dataset)):
        ep = dataset[i]
        uid = str(ep.meta.get('uid', f'ts-{ep.meta.get("created_ts_ns", i)}'))
        # ``eval.success`` is the env's task-success verdict; ``eval.terminated`` only means the episode
        # ended within budget, so a failed-but-done rollout is terminated=True, success=False and a
        # timed-out one is terminated=False with no success verdict. Keep both so the roll-up can tell a
        # scored failure from an unscored episode.
        verdict = ep.static.get('eval.success')
        terminated = ep.static.get('eval.terminated')
        facts[uid] = _RecordedFacts(
            duration_s=ep.duration_ns / 1e9,
            size_mb=float(ep.meta.get('size_mb', 0.0)),
            success=None if verdict is None else bool(verdict),
            terminated=None if terminated is None else bool(terminated),
            eval_key=str(ep.static.get('eval.embodiment', '')),
        )
    return facts


def _parse_dmon(log_path: Path) -> GpuSummary:
    """Mean SM utilisation and peak framebuffer (GB) from an ``nvidia-smi dmon`` log.

    Column layout varies — ``-o DT`` prepends date/time, ``-s u`` adds encoder/decoder (and, on newer
    drivers, JPEG/OFA) columns before the framebuffer — so the positions are read from the ``# ... sm ...
    fb ...`` name header rather than hard-coded. ``sm`` is SM utilisation (%) and ``fb`` the framebuffer
    use (MiB). Rows before the header, or with missing numeric fields, are skipped rather than failing.
    """
    sm_idx: int | None = None
    fb_idx: int | None = None
    utils: list[float] = []
    fb_mib: list[float] = []
    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            # The name header carries both 'sm' and 'fb'; the units line ('%', 'MB') does not.
            names = stripped.lstrip('#').split()
            if 'sm' in names and 'fb' in names:
                sm_idx, fb_idx = names.index('sm'), names.index('fb')
            continue
        if sm_idx is None or fb_idx is None:
            continue
        fields = line.split()
        try:
            utils.append(float(fields[sm_idx]))
            fb_mib.append(float(fields[fb_idx]))
        except (IndexError, ValueError):
            continue
    return GpuSummary(
        mean_util_pct=float(np.mean(utils)) if utils else 0.0, peak_vram_gb=(max(fb_mib) / 1024.0) if fb_mib else 0.0
    )


def _success_outcome(facts: _RecordedFacts, has_oracle: bool) -> bool | None:
    """Whether the episode is a success (True), a failure (False), or unscored (None).

    A recorded ``eval.success`` is taken as-is. With none, an episode that reached a verdict or timed out
    (``eval.terminated`` recorded) under an eval that *does* score success elsewhere is a failure — this is
    the timed-out rollout an adapter only marks on live success. With no success oracle anywhere in the
    run there is nothing to score.
    """
    if facts.success is not None:
        return facts.success
    if has_oracle and facts.terminated is not None:
        return False
    return None


def _build_report(records: list[dict], facts: dict[str, _RecordedFacts], gpu: dict[str, GpuSummary]) -> PassReport:
    wall = np.array([r['wall_s'] for r in records], dtype=float)
    wall_pass = float(wall.sum())
    all_infer_ms = np.array([ms for r in records for ms in r['infer_ms']], dtype=float)
    phases = ('reset_s', 'env_step_s', 'policy_wait_s', 'record_io_s', 'overhead_s')
    # Aggregate fraction = summed phase over summed wall, so long episodes weigh proportionally.
    split_fractions = {phase: float(np.sum([r[phase] for r in records]) / wall_pass) for phase in phases}

    matched = [facts[r['episode_uid']] for r in records if r['episode_uid'] in facts]
    sim_seconds = sum(f.duration_s for f in matched)
    # The success oracle is per eval, not pass-wide: an eval that records no ``eval.success`` anywhere has no
    # verdict to fail against, so its episodes stay unscored even when another eval in the same sweep does score
    # successes. Group by the recorded eval identity (embodiment descriptor) — a scored and an unscored eval
    # always differ there, unlike the optional ``eval.task`` label — so an unscored eval can't inherit a scored
    # one's oracle and have its timeouts counted as failures.
    by_eval: dict[str, list[_RecordedFacts]] = defaultdict(list)
    for f in matched:
        by_eval[f.eval_key].append(f)
    judged = [
        outcome
        for group in by_eval.values()
        for outcome in (_success_outcome(f, any(g.success is not None for g in group)) for f in group)
        if outcome is not None
    ]
    return PassReport(
        episodes=len(records),
        wall_pass_s=wall_pass,
        real_time_factor=(sim_seconds / wall_pass) if wall_pass else 0.0,
        policy_busy_fraction=float(np.sum([r['policy_wait_s'] for r in records]) / wall_pass) if wall_pass else 0.0,
        infer_calls=int(all_infer_ms.size),
        infer_p50_ms=float(np.percentile(all_infer_ms, 50)) if all_infer_ms.size else 0.0,
        infer_p95_ms=float(np.percentile(all_infer_ms, 95)) if all_infer_ms.size else 0.0,
        mean_split_fractions=split_fractions,
        mean_bytes_per_rollout=(sum(f.size_mb for f in matched) / len(matched) * 1024 * 1024) if matched else 0.0,
        success_rate=(sum(judged) / len(judged)) if judged else None,
        gpu=gpu,
    )


def _render(report: PassReport) -> str:
    lines = [
        f'episodes:            {report.episodes}',
        f'W_pass (wall):       {report.wall_pass_s:.1f} s ({report.wall_pass_s / 3600:.2f} h)',
        f'real-time factor:    {report.real_time_factor:.3f} (sim-s per wall-s)',
        f'policy busy (k):     {report.policy_busy_fraction:.3f}  -> ~{1 / report.policy_busy_fraction:.1f} sims/H100'
        if report.policy_busy_fraction
        else 'policy busy (k):     n/a',
        f'infer calls:         {report.infer_calls}',
        f'infer p50 / p95:     {report.infer_p50_ms:.1f} / {report.infer_p95_ms:.1f} ms',
        f'bytes / rollout:     {report.mean_bytes_per_rollout / 1e6:.1f} MB',
        f'success rate:        {report.success_rate:.3f}'
        if report.success_rate is not None
        else 'success rate:        n/a',
        'wall split (fraction of W_pass):',
    ]
    lines += [f'  {phase[:-2]:<12} {frac:.3f}' for phase, frac in report.mean_split_fractions.items()]
    for box, summary in report.gpu.items():
        lines.append(f'gpu[{box}]: util {summary.mean_util_pct:.0f}%  peak VRAM {summary.peak_vram_gb:.1f} GB')
    return '\n'.join(lines)


@cfn.config(gpu_sim_log=None, gpu_policy_log=None)
def timing_report(dataset_dir: str, gpu_sim_log: str | None, gpu_policy_log: str | None):
    """Reduce ``<dataset_dir>/timing.jsonl`` against the recorded episodes into a pass report.

    ``dataset_dir`` may be an ``s3://`` URI — the documented Nebius eval path writes there — in which case
    it is pulled local first, mirroring how the eval command syncs its output. ``gpu_sim_log`` /
    ``gpu_policy_log`` are optional ``nvidia-smi dmon`` logs (sim box / policy endpoint) whose utilisation
    and peak VRAM are folded in. Writes ``timing_summary.json`` alongside the input and prints the report.
    """
    # Pull a remote dataset local before reading; a plain local path is used as-is (never handed to pos3,
    # whose download would prune it).
    root = Path(pos3.download(dataset_dir)) if '://' in dataset_dir else Path(dataset_dir)
    timing_path = root / TIMING_FILENAME
    records = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f'no timing records in {timing_path}')

    # The sim box samples its own GPU into the dataset dir under ``--timing``; an explicit path overrides
    # it, and the policy endpoint's log (a different box) is only ever passed in.
    gpu: dict[str, GpuSummary] = {}
    sim_log = Path(gpu_sim_log) if gpu_sim_log is not None else root / GPU_LOG_FILENAME
    if sim_log.exists():
        gpu['sim'] = _parse_dmon(sim_log)
    if gpu_policy_log is not None:
        gpu['policy'] = _parse_dmon(Path(gpu_policy_log))

    report = _build_report(records, _recorded_facts(root), gpu)
    summary_path = root / 'timing_summary.json'
    summary_path.write_text(json.dumps({**report.__dict__, 'gpu': {k: v.__dict__ for k, v in gpu.items()}}, indent=2))
    logger.info(f'wrote {summary_path}')
    print(_render(report))
