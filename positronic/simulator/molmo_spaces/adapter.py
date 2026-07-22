"""Bridge a pi05_droid policy served over positronic's inference protocol into a MolmoSpaces eval.

MolmoSpaces drives a DROID FR3 + Robotiq rig and expects a ``BasePolicy`` whose ``get_action`` maps its
observation dict to a per-move-group action dict. positronic serves pi05_droid behind an ``InferenceServer``
(FastAPI on port 8000) whose DROID codec consumes the raw positronic observation keys
(``robot_state.q``, ``grip``, ``image.wrist``, ``image.exterior``, ``task``) and returns a chunk of decoded
per-step ``JointDelta`` commands (7 joint velocities already scaled by ``MAX_JOINT_DELTA``) plus a binarized grip.

This module holds:
  - two pure mapping functions — ``molmo_obs_to_positronic`` and ``positronic_action_to_molmo`` — that carry the
    whole translation and are import-free of both frameworks so they run under a bare pytest;
  - ``ChunkBuffer``, which plays one buffered chunk step per policy tick and re-queries when the chunk drains,
    matching DROID's open-loop horizon;
  - ``FakePolicy``, a server-free stand-in honoring the positronic client contract, for smoke runs;
  - ``MolmoSpacesPolicy``, the ``InferencePolicy`` subclass wiring the above into MolmoSpaces (only usable where
    molmo_spaces is installed; the mapping logic above is not).

MolmoSpaces and positronic are imported behind guards so the pure logic and its tests need neither installed.
"""

import collections.abc as cabc
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image as PilImage

try:
    from molmo_spaces.policy.base_policy import InferencePolicy as _InferencePolicyBase

    HAS_MOLMO_SPACES = True
except ImportError:
    _InferencePolicyBase = object
    HAS_MOLMO_SPACES = False


# MolmoSpaces DROID rig observation keys (FrankaDroidCameraSystem + RobotJointPositionSensor).
MOLMO_WRIST_CAMERA = 'wrist_camera'
MOLMO_EXTERIOR_CAMERA = 'exo_camera_1'
MOLMO_ARM_GROUP = 'arm'
MOLMO_GRIPPER_GROUP = 'gripper'

# positronic raw observation keys the DROID codec (positronic/vendors/openpi/codecs.py:droid_obs) reads.
POS_JOINTS = 'robot_state.q'
POS_GRIP = 'grip'
POS_WRIST_IMAGE = 'image.wrist'
POS_EXTERIOR_IMAGE = 'image.exterior'
POS_TASK = 'task'

# The model consumes 224x224 RGB (openpi DROID preprocessing); the DROID codec re-pads to this, so matching it
# here makes the server-side resize a no-op and keeps the wire payload small.
IMAGE_SIZE = (224, 224)
NUM_ARM_JOINTS = 7

# Robotiq closure at which the FR3 gripper qpos saturates; the pi baseline normalizes proprio grip by it
# (molmospaces pi_policy.py:126) so the served model sees the [0, 1] closure it was trained on.
GRIPPER_QPOS_CLOSED = 0.824033

# Robotiq 2F-85 control convention: 0 fully open, 255 fully closed (franka_droid_view.py:43).
ROBOTIQ_OPEN = 0.0
ROBOTIQ_CLOSED = 255.0

# DROID re-queries after an 8-step open-loop horizon (codecs.py:172); mirror it in the fake client.
DROID_CHUNK_STEPS = 8

# JointDelta scales the clipped [-1, 1] velocities by this (positronic policy/action.py:194).
MAX_JOINT_DELTA = 0.2


def resize_with_pad(image: np.ndarray, width: int, height: int, resample=PilImage.Resampling.BILINEAR) -> np.ndarray:
    """Aspect-preserving resize into a ``height x width`` frame, zero-padded — positronic's DROID preprocessing.

    Reproduces ``positronic.dataset.transforms.image.resize_with_pad_per_frame``: scale the longer side to fit,
    then center the result on a black canvas. An already-correctly-sized frame passes through untouched.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected an HWC RGB frame, got shape {image.shape}')
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.shape[0] == height and image.shape[1] == width:
        return image

    pil = PilImage.fromarray(image)
    cur_width, cur_height = pil.size
    ratio = max(cur_width / width, cur_height / height)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)
    resized = pil.resize((resized_width, resized_height), resample=resample)

    canvas = PilImage.new(resized.mode, (width, height), 0)
    canvas.paste(resized, (max(0, (width - resized_width) // 2), max(0, (height - resized_height) // 2)))
    return np.asarray(canvas)


def _single_env(observation: Any) -> dict:
    """MolmoSpaces yields ``observation`` as a list (one dict per batch env); single-env eval uses index 0."""
    if isinstance(observation, cabc.Mapping):
        return observation
    if isinstance(observation, cabc.Sequence):
        return observation[0]
    raise TypeError(f'Unexpected observation container: {type(observation)}')


def molmo_obs_to_positronic(
    observation: Any,
    task: str,
    *,
    wrist_key: str = MOLMO_WRIST_CAMERA,
    exterior_key: str = MOLMO_EXTERIOR_CAMERA,
    image_size: tuple[int, int] = IMAGE_SIZE,
    gripper_qpos_closed: float = GRIPPER_QPOS_CLOSED,
) -> dict[str, Any]:
    """Map a MolmoSpaces DROID observation to the raw positronic keys the DROID codec consumes.

    ``observation`` may be the batch list or a single env dict. The arm joints pass through as 7 absolute radians;
    the Robotiq finger qpos is normalized to the ``[0, 1]`` closure the model was trained on; both cameras are
    resized-with-pad to ``image_size``.
    """
    env = _single_env(observation)
    qpos = env['qpos']
    arm = np.asarray(qpos['arm'], dtype=np.float32).reshape(-1)
    grip_qpos = float(np.asarray(qpos['gripper']).reshape(-1)[0])
    grip = float(np.clip(grip_qpos / gripper_qpos_closed, 0.0, 1.0))

    width, height = image_size
    return {
        POS_JOINTS: arm,
        POS_GRIP: np.array([grip], dtype=np.float32),
        POS_WRIST_IMAGE: resize_with_pad(env[wrist_key], width, height),
        POS_EXTERIOR_IMAGE: resize_with_pad(env[exterior_key], width, height),
        POS_TASK: task,
    }


def _wire_get(mapping: cabc.Mapping, name: str, default: Any = None) -> Any:
    """Read a key that may arrive str- or bytes-keyed: msgpack deserialisation yields bytes keys on some
    positronic/msgpack version combinations, so both spellings must be tried."""
    for key in (name, name.encode()):
        if key in mapping:
            return mapping[key]
    return default


def _joint_delta_velocities(robot_command: Any) -> np.ndarray:
    """Read the 7 joint velocities from a decoded positronic ``JointDelta`` (object) or its wire dict."""
    if hasattr(robot_command, 'velocities'):
        return np.asarray(robot_command.velocities, dtype=np.float32).reshape(-1)
    if isinstance(robot_command, cabc.Mapping):
        velocities = _wire_get(robot_command, 'velocities')
        if velocities is not None:
            return np.asarray(velocities, dtype=np.float32).reshape(-1)
    raise TypeError(f'Cannot read joint velocities from {type(robot_command)}')


def positronic_action_to_molmo(
    action: cabc.Mapping,
    current_arm_qpos: np.ndarray,
    *,
    arm_group: str = MOLMO_ARM_GROUP,
    gripper_group: str = MOLMO_GRIPPER_GROUP,
) -> dict[str, np.ndarray]:
    """Turn one decoded positronic DROID action into a MolmoSpaces per-move-group action.

    The server returns a per-step ``JointDelta`` (velocities relative to the live measured joints, DROID's control
    convention) plus a binarized grip. Integrating each delta onto the joints measured this tick reproduces the
    positronic driver's ``set_target_joints(q + delta)``; the grip maps to the Robotiq open/closed control values.
    """
    robot_command = _wire_get(action, 'robot_command')
    if robot_command is None:
        raise KeyError(f'no robot_command in action; keys={list(action.keys())}')
    velocities = _joint_delta_velocities(robot_command)
    current = np.asarray(current_arm_qpos, dtype=np.float32).reshape(-1)
    if velocities.shape[0] != current.shape[0]:
        raise ValueError(f'Joint count mismatch: delta {velocities.shape[0]} vs qpos {current.shape[0]}')

    grip = float(_wire_get(action, 'target_grip', 0.0))
    gripper = ROBOTIQ_CLOSED if grip > 0.5 else ROBOTIQ_OPEN
    return {arm_group: (current + velocities).astype(np.float32), gripper_group: np.array([gripper], dtype=np.float32)}


class ChunkBuffer:
    """Plays one action per tick from a buffered inference chunk, re-querying the session when it drains.

    The positronic session returns a whole action chunk per call; MolmoSpaces asks for one action per policy step.
    Buffering here re-queries every ``len(chunk)`` steps, matching DROID's open-loop horizon.
    """

    def __init__(self, session: Any):
        self._session = session
        self._pending: list[Any] = []

    def next(self, obs: dict[str, Any]) -> Any:
        if not self._pending:
            chunk = self._session(obs)
            # The serving layer ends each chunk with a horizon marker carrying only `timestamp`
            # (droid's action window is horizon=8/15: 8 actions + the window-end stamp). It is not
            # an action — playing it as one KeyErrors on `robot_command`, so keep only action-bearing entries.
            self._pending = [c for c in (chunk or []) if _wire_get(c, 'robot_command') is not None]
        if not self._pending:
            raise RuntimeError('Inference session returned an empty action chunk')
        return self._pending.pop(0)

    def reset(self) -> None:
        self._pending = []

    def close(self) -> None:
        close = getattr(self._session, 'close', None)
        if close is not None:
            close()


@dataclass
class _FakeJointDelta:
    """Duck-typed stand-in for positronic ``command.JointDelta`` (read via ``.velocities``)."""

    TYPE = 'joint_delta'
    velocities: np.ndarray


class FakeSession:
    """Server-free session emitting action chunks in the shape a positronic ``RemoteSession`` returns."""

    def __init__(
        self, *, chunk_size: int, num_joints: int, max_joint_delta: float, mode: str, rng: np.random.Generator
    ):
        self._chunk_size = chunk_size
        self._num_joints = num_joints
        self._max_joint_delta = max_joint_delta
        self._mode = mode
        self._rng = rng

    def _velocities(self) -> np.ndarray:
        if self._mode == 'zero':
            return np.zeros(self._num_joints, dtype=np.float32)
        return (self._rng.uniform(-1.0, 1.0, self._num_joints) * self._max_joint_delta).astype(np.float32)

    def _grip(self) -> float:
        if self._mode == 'zero':
            return 0.0
        return 1.0 if self._rng.random() > 0.5 else 0.0

    def __call__(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {'robot_command': _FakeJointDelta(self._velocities()), 'target_grip': self._grip()}
            for _ in range(self._chunk_size)
        ]

    @property
    def meta(self) -> dict[str, Any]:
        return {'type': 'fake'}

    def close(self) -> None:
        return None


class FakePolicy:
    """Drop-in for positronic ``RemotePolicy`` that needs no server — random or zero DROID action chunks.

    ``mode='random'`` draws velocities uniformly in ``[-max_joint_delta, max_joint_delta]`` and a random binary
    grip; ``mode='zero'`` holds the arm and keeps the gripper open. Sessions are deterministic given ``seed``.
    """

    def __init__(
        self,
        *,
        chunk_size: int = DROID_CHUNK_STEPS,
        num_joints: int = NUM_ARM_JOINTS,
        max_joint_delta: float = MAX_JOINT_DELTA,
        mode: str = 'random',
        seed: int = 0,
    ):
        if mode not in ('random', 'zero'):
            raise ValueError(f"mode must be 'random' or 'zero', got {mode!r}")
        self._chunk_size = chunk_size
        self._num_joints = num_joints
        self._max_joint_delta = max_joint_delta
        self._mode = mode
        self._seed = seed
        self._session_count = 0

    def new_session(self, context: dict[str, Any] | None = None) -> FakeSession:
        rng = np.random.default_rng(self._seed + self._session_count)
        self._session_count += 1
        return FakeSession(
            chunk_size=self._chunk_size,
            num_joints=self._num_joints,
            max_joint_delta=self._max_joint_delta,
            mode=self._mode,
            rng=rng,
        )

    @property
    def meta(self) -> dict[str, Any]:
        return {'type': 'fake', 'mode': self._mode}

    def close(self) -> None:
        return None


def make_policy_client(
    *,
    fake: bool = False,
    host: str = 'localhost',
    port: int = 8000,
    model_id: str | None = None,
    secure: bool = False,
    fake_mode: str = 'random',
    fake_seed: int = 0,
    fake_chunk_size: int = DROID_CHUNK_STEPS,
) -> Any:
    """Return the inference client the policy talks to: a ``FakePolicy`` when ``fake`` else positronic ``RemotePolicy``.

    ``RemotePolicy`` is imported lazily so this module (and its tests) load without positronic installed.
    """
    if fake:
        return FakePolicy(mode=fake_mode, seed=fake_seed, chunk_size=fake_chunk_size)
    from positronic.policy.remote import RemotePolicy

    return RemotePolicy(host, port=port, model_id=model_id, secure=secure)


@dataclass
class AdapterConfig:
    """Static rig/codec wiring for the adapter, independent of any framework object."""

    wrist_key: str = MOLMO_WRIST_CAMERA
    exterior_key: str = MOLMO_EXTERIOR_CAMERA
    arm_group: str = MOLMO_ARM_GROUP
    gripper_group: str = MOLMO_GRIPPER_GROUP
    image_size: tuple[int, int] = IMAGE_SIZE
    gripper_qpos_closed: float = GRIPPER_QPOS_CLOSED


class MolmoSpacesPolicy(_InferencePolicyBase):
    """MolmoSpaces ``InferencePolicy`` serving pi05_droid through a positronic inference client.

    Wires the pure mapping functions and ``ChunkBuffer`` into the ``obs_to_model_input`` → ``inference_model`` →
    ``model_output_to_action`` pipeline. Instantiable only where molmo_spaces is installed; the mapping logic it
    delegates to is not, and is unit-tested directly.
    """

    def __init__(
        self,
        config: Any = None,
        task: Any = None,
        *,
        client: Any = None,
        prompt: str = '',
        adapter_config: AdapterConfig | None = None,
        client_kwargs: dict[str, Any] | None = None,
    ):
        if HAS_MOLMO_SPACES:
            super().__init__(config, task)
        self._client = client if client is not None else make_policy_client(**(client_kwargs or {}))
        self._prompt = prompt
        self._adapter = adapter_config or AdapterConfig()
        self._buffer: ChunkBuffer | None = None
        self._task_text = prompt
        self._current_arm_qpos = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)

    def prepare_model(self) -> None:
        self._open_session()

    def reset(self) -> None:
        task = getattr(self, 'task', None)
        self._task_text = task.get_task_description() if task is not None else self._prompt
        self._open_session()

    def _open_session(self) -> None:
        if self._buffer is not None:
            self._buffer.close()
        self._buffer = ChunkBuffer(self._client.new_session())

    def close(self) -> None:
        """Close the open inference session. MolmoSpaces' runner never calls this itself, so the caller closes
        each episode's policy explicitly — otherwise every per-episode policy leaks one live server session
        (each of which also resets a serverless endpoint's idle clock)."""
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None

    def obs_to_model_input(self, observation: Any) -> dict[str, Any]:
        env = _single_env(observation)
        self._current_arm_qpos = np.asarray(env['qpos']['arm'], dtype=np.float32).reshape(-1)
        return molmo_obs_to_positronic(
            env,
            self._task_text,
            wrist_key=self._adapter.wrist_key,
            exterior_key=self._adapter.exterior_key,
            image_size=self._adapter.image_size,
            gripper_qpos_closed=self._adapter.gripper_qpos_closed,
        )

    def inference_model(self, model_input: dict[str, Any]) -> Any:
        if self._buffer is None:
            self._open_session()
        return self._buffer.next(model_input)

    def model_output_to_action(self, model_output: cabc.Mapping) -> dict[str, np.ndarray]:
        return positronic_action_to_molmo(
            model_output,
            self._current_arm_qpos,
            arm_group=self._adapter.arm_group,
            gripper_group=self._adapter.gripper_group,
        )

    def get_info(self) -> dict[str, Any]:
        info = super().get_info() if HAS_MOLMO_SPACES else {}
        info['client'] = getattr(self._client, 'meta', {})
        return info
