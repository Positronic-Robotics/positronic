"""Exercise the ACT demo through the eval CLI and check its recorded behavior."""

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, urlunsplit
from zipfile import ZIP_LZMA, ZipFile

import configuronic as cfn
import mujoco as mj
import numpy as np

from positronic import keys
from positronic.cfg.simulator import STACK_GREEN_CUBE, STACK_RED_CUBE
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.simulator.mujoco.transforms import load_spec
from positronic.utils import package_assets_path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = Path(__file__).resolve().parent / 'fixtures' / 'act_stack'
SEEDS = (4, 8)
CHECKPOINT_ID = '050000'
CHECKPOINT_PATH = (
    f's3://PUBLIC@positronic-public/checkpoints/sim_stack_cubes/act/checkpoints/{CHECKPOINT_ID}/pretrained_model'
)
EPISODE_SECONDS = 15
HOLD_NS = 500_000_000
STATE_SPEC = mj.mjtState.mjSTATE_INTEGRATION
FINGER_BODIES = ('left_finger_ph', 'right_finger_ph')
CUBE_POSES = 'cube_poses'
SUPPORTED = 'stack_supported'
TIME_SUFFIX = '.time_ns'
ROBOT_OBSERVATIONS = (keys.EE_POSE, keys.JOINTS, keys.GRIP)
RECORDED_SIGNALS = (keys.TARGET_EE_POSE, keys.TARGET_GRIP, *ROBOT_OBSERVATIONS)
SEED_DIRECTORY = 'seed_{seed}'
REFERENCE_FILENAME = SEED_DIRECTORY + '.npz'


def checkpoint_url(url: str) -> str:
    parsed = urlsplit(url if '://' in url else f'http://{url}')
    if not parsed.netloc or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        raise ValueError('--url must name the server origin only, e.g. http://localhost:8000')
    return urlunsplit((parsed.scheme, parsed.netloc, f'/api/v1/session/{CHECKPOINT_ID}', '', ''))


def run_episode(url: str, output: Path, seed: int, wall_timeout: float) -> None:
    command = [
        str(Path(sys.executable).with_name('positronic')),
        'eval',
        'run',
        '--eval=.sim.positronic.stack_cubes',
        '--policy=.remote',
        f'--policy.url={checkpoint_url(url)}',
        f'--eval.seed={seed}',
        '--eval.trial_count=1',
        f'--eval.timeout={EPISODE_SECONDS}',
        '--charge_inference_time=False',
        f'--output_dir={output}',
    ]
    log = output.with_suffix('.log')
    print(f'Seed {seed}: running; log: {log}', flush=True)
    with log.open('w') as stream:
        subprocess.run(
            command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=wall_timeout
        )


def read_episode(output: Path, seed: int) -> Episode:
    dataset = LocalDataset(output)
    if len(dataset) != 1:
        raise ValueError(f'{output}: expected one completed episode, found {len(dataset)}')
    episode = next(iter(dataset))
    expected = {
        eval_keys.SEED: seed,
        eval_keys.TIMEOUT: EPISODE_SECONDS,
        eval_keys.CHARGE_INFERENCE_TIME: False,
        eval_keys.TERMINATED: False,
        f'{policy_keys.SERVER_META}.{policy_keys.CHECKPOINT_PATH}': CHECKPOINT_PATH,
    }
    for name, value in expected.items():
        if episode.static[name] != value:
            raise ValueError(f'Seed {seed}: {name} is {episode.static[name]!r}, expected {value!r}')
    return episode


def is_supported_stack(model: mj.MjModel, data: mj.MjData, red: int, green: int, fingers: set[int]) -> bool:
    touching: set[int] = set()
    for contact in data.contact:
        bodies = {int(model.geom_bodyid[g]) for g in contact.geom}
        if green in bodies and contact.efc_address >= 0:
            touching.update(bodies - {green})
    height = data.xpos[green, 2] - data.xpos[red, 2]
    return red in touching and not (touching & fingers) and 0.015 < height < 0.025


def cube_trace(
    episode: Episode, scene_key: str = 'scene_xml', state_key: str = f'sim_state.{STATE_SPEC.name}'
) -> dict[str, np.ndarray]:
    spec, _ = load_spec(episode.static[scene_key], Path(package_assets_path('assets/mujoco')))
    model = spec.compile()
    data = mj.MjData(model)
    red, green = model.body(STACK_RED_CUBE.body_name).id, model.body(STACK_GREEN_CUBE.body_name).id
    fingers = {model.body(name).id for name in FINGER_BODIES}
    state_signal = episode[state_key]
    times = np.asarray(list(state_signal.keys()), dtype=np.int64) - episode.start_ts
    if np.any(np.diff(times) > np.ceil(model.opt.timestep * 1e9) + 1):
        raise ValueError(f'{state_key}: recording skips physics steps')
    poses = np.empty((len(times), 2, 7))
    supported = np.empty(len(times), dtype=bool)
    for i, state in enumerate(state_signal.values()):
        if not np.isfinite(state).all():
            raise ValueError(f'{state_key}: non-finite state at {times[i] / 1e9:.6f}s')
        mj.mj_setState(model, data, state, STATE_SPEC)
        mj.mj_forward(model, data)
        poses[i, :, :3] = data.xpos[[red, green]]
        poses[i, :, 3:] = data.xquat[[red, green]]
        supported[i] = is_supported_stack(model, data, red, green, fingers)
    return {CUBE_POSES: poses, CUBE_POSES + TIME_SUFFIX: times, SUPPORTED: supported, SUPPORTED + TIME_SUFFIX: times}


def read_trace(episode: Episode) -> dict[str, np.ndarray]:
    trace = cube_trace(episode)
    for name in RECORDED_SIGNALS:
        signal = episode[name]
        trace[name] = np.asarray(list(signal.values()))
        trace[name + TIME_SUFFIX] = np.asarray(list(signal.keys()), dtype=np.int64) - episode.start_ts
    for name, values in trace.items():
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f'{name}: empty or non-finite recording')
        if name.endswith(TIME_SUFFIX) and np.any(np.diff(values) <= 0):
            raise ValueError(f'{name}: timestamps must increase strictly')
    for name in ROBOT_OBSERVATIONS:
        if not np.array_equal(trace[name + TIME_SUFFIX], trace[CUBE_POSES + TIME_SUFFIX]):
            raise ValueError(f'{name}: expected one observation at every recorded physics step')
    return trace


def check_stacking(trace: Mapping[str, np.ndarray]) -> float:
    times = trace[SUPPORTED + TIME_SUFFIX]
    supported = trace[SUPPORTED]
    if not len(times) or len(times) != len(supported):
        raise ValueError('Stacking check needs a nonempty support signal with matching timestamps')
    if times[0] > 0 or times[-1] < (EPISODE_SECONDS - 0.01) * 1e9:
        raise ValueError('Recording does not cover the full episode')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Stacking timestamps must increase strictly')
    start = None
    for timestamp, supported_now in zip(times, supported, strict=True):
        if not supported_now:
            start = None
        elif start is None:
            start = timestamp
        elif timestamp - start >= HOLD_NS:
            return float(timestamp / 1e9)
    raise ValueError('Green never rested on red without gripper contact for 0.5 seconds')


def compare_trace(actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]) -> None:
    if set(actual) != set(expected):
        raise ValueError(f'Trace fields differ: {sorted(set(actual) ^ set(expected))}')
    for name in sorted(expected):
        got, want = actual[name], expected[name]
        if got.shape != want.shape:
            raise ValueError(f'{name}: shape {got.shape}, expected {want.shape}')
        if not np.isfinite(got).all() or not np.isfinite(want).all():
            raise ValueError(f'{name}: non-finite trace')
        differing = np.argwhere(got != want)
        if differing.size:
            index = tuple(int(i) for i in differing[0])
            time_key = name if name.endswith(TIME_SUFFIX) else name + TIME_SUFFIX
            time = expected[time_key][index[0]] / 1e9
            raise ValueError(
                f'{name}: first difference at {time:.9f}s, index {index}: {got[index]!r}, expected {want[index]!r}'
            )


def check_episode(output: Path, seed: int, reference: Path, success_only: bool) -> dict[str, np.ndarray]:
    trace = read_trace(read_episode(output, seed))
    completed_at = check_stacking(trace)
    if not success_only:
        with np.load(reference / REFERENCE_FILENAME.format(seed=seed), allow_pickle=False) as expected:
            compare_trace(trace, expected)
    result = 'stacking passed' if success_only else 'stacking and exact trace passed'
    print(f'Seed {seed}: {result}; stacked by {completed_at:.3f}s', flush=True)
    return trace


def check_seeds(seeds: Sequence[int]) -> None:
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError(f'Choose distinct seeds from {SEEDS}, got {seeds}')


@cfn.config(seeds=SEEDS, reference_dir=str(REFERENCE_DIR), success_only=False, wall_timeout=120.0)
def run(url: str, output_dir: str, seeds: Sequence[int], reference_dir: str, success_only: bool, wall_timeout: float):
    """Run the ACT stacking episodes, retaining their recordings and logs."""
    check_seeds(seeds)
    if not np.isfinite(wall_timeout) or wall_timeout <= 0:
        raise ValueError('wall_timeout must be finite and positive')
    output_root, reference = Path(output_dir).expanduser().resolve(), Path(reference_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    for seed in seeds:
        output = output_root / SEED_DIRECTORY.format(seed=seed)
        run_episode(url, output, seed, wall_timeout)
        check_episode(output, seed, reference, success_only)


@cfn.config(seeds=SEEDS, reference_dir=str(REFERENCE_DIR), success_only=False)
def check(output_dir: str, seeds: Sequence[int], reference_dir: str, success_only: bool):
    """Check recorded episodes without running the model."""
    check_seeds(seeds)
    output_root, reference = Path(output_dir).expanduser(), Path(reference_dir).expanduser()
    for seed in seeds:
        check_episode(output_root / SEED_DIRECTORY.format(seed=seed), seed, reference, success_only)


def write_npz(path: Path, trace: Mapping[str, np.ndarray]) -> None:
    with ZipFile(path, 'w', compression=ZIP_LZMA) as archive:
        for name, values in trace.items():
            with archive.open(f'{name}.npy', 'w') as stream:
                np.save(stream, values, allow_pickle=False)


@cfn.config(seeds=SEEDS)
def capture(output_dir: str, reference_dir: str, seeds: Sequence[int]):
    """Capture successful recorded behavior into a new reference directory."""
    check_seeds(seeds)
    output_root, reference = Path(output_dir).expanduser(), Path(reference_dir).expanduser()
    if reference.exists() or reference.is_symlink():
        raise FileExistsError(reference)
    reference.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=reference.parent, prefix=f'.{reference.name}-') as staging_dir:
        staging = Path(staging_dir)
        for seed in seeds:
            trace = check_episode(output_root / SEED_DIRECTORY.format(seed=seed), seed, reference, success_only=True)
            write_npz(staging / REFERENCE_FILENAME.format(seed=seed), trace)
        # rename can replace an empty directory; mkdir reserves a new destination.
        reference.mkdir()
        try:
            staging.rename(reference)
        except OSError:
            reference.rmdir()
            raise
    print(f'References written to {reference}; record their provenance before committing them.')


def main() -> int:
    try:
        cfn.cli({'run': run, 'check': check, 'capture': capture})
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f'FAIL: {exc}\nRecordings and logs remain in the requested output_dir.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
