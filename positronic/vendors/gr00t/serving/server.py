"""Serve one native GR00T N1.7 DROID checkpoint in its own Python environment."""

import json
import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import pos3
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY
from positronic_model_server.server import Model, ModelServer, Session
from positronic_model_server.server_wire import ServedHostPort
from positronic_model_server.websocket_wire import WebsocketWire

from . import (
    ACTION,
    BASE_MODEL,
    CHECKPOINT_PREFIX,
    CONFIG_FILENAME,
    EE_POSE,
    EMBODIMENT,
    GRIP,
    HF_MODEL_PREFIX,
    JOINT_POSITION,
    LANGUAGE,
    OPTIMIZER_FILENAME,
    STATE,
    STATE_DIMS,
    TASK,
    VIDEO,
    keys,
    recipe,
)
from . import settings as model_settings

logger = logging.getLogger(__name__)
AUTH_TOKEN_ENV = 'AUTH_TOKEN'
MODEL_SOURCE_ENV = 'GROOT_MODEL_SOURCE'


def _select_checkpoint_name(children: list[str], checkpoint: int | None, source: str) -> str:
    steps = sorted(
        (int(name.removeprefix(CHECKPOINT_PREFIX)), name)
        for name in children
        if name.startswith(CHECKPOINT_PREFIX) and name.removeprefix(CHECKPOINT_PREFIX).isdigit()
    )
    if not steps:
        raise ValueError(f'No GR00T checkpoints in {source}. Available: {children}')
    if checkpoint is None:
        return steps[-1][1]
    for step, name in steps:
        if step == checkpoint:
            return name
    raise ValueError(f'Checkpoint not found: {checkpoint}. Available: {[name for _, name in steps]}')


@pos3.mirror()
def resolve_checkpoint(
    model_source: str, checkpoint: int | None = None, revision: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Resolve a local, S3 or Hugging Face source to one immutable selection for this process."""
    source = model_source.rstrip('/')
    if source.startswith(HF_MODEL_PREFIX):
        # Local/S3 resolution and CLI help do not require the model's Hub client.
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        if checkpoint is not None:
            raise ValueError('checkpoint step selection applies only to experiment directories')
        repo = source.removeprefix(HF_MODEL_PREFIX)
        path = Path(snapshot_download(repo, revision=revision))
        return path, {
            keys.CHECKPOINT_ID: repo,
            keys.CHECKPOINT_PATH: str(path),
            keys.EXPERIMENT_NAME: repo.rsplit('/', 1)[-1],
            'checkpoint_revision': path.name,
        }
    if revision is not None:
        raise ValueError('revision applies only to Hugging Face checkpoints')
    if not source.startswith('s3://'):
        source = str(Path(source).expanduser().resolve())
    children = [child.rstrip('/').rsplit('/', 1)[-1] for child in pos3.ls(source, recursive=False)]
    experiment_name = source.rsplit('/', 1)[-1]
    if CONFIG_FILENAME in children:
        if checkpoint is not None:
            raise ValueError('checkpoint step selection needs an experiment directory, not a checkpoint')
        selected = source
        if experiment_name.startswith(CHECKPOINT_PREFIX):
            experiment_name = source.rsplit('/', 2)[-2]
    else:
        selected = f'{source}/{_select_checkpoint_name(children, checkpoint, source)}'
    suffix = selected.rsplit('/', 1)[-1].removeprefix(CHECKPOINT_PREFIX)
    path = Path(pos3.download(selected, exclude=[OPTIMIZER_FILENAME]))
    return path, {
        keys.CHECKPOINT_ID: str(int(suffix)) if suffix.isdigit() else suffix,
        keys.CHECKPOINT_PATH: selected,
        keys.EXPERIMENT_NAME: experiment_name,
    }


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


class Gr00tSession(Session):
    def __init__(self, policy: Any, client_stack: dict[str, Any], rtc: dict[str, float] | None = None, *, fps: float):
        super().__init__(client_stack)
        self._policy = policy
        self._rtc = rtc
        self._fps = fps
        self._previous_poses: np.ndarray | None = None

    def _rtc_options(self, context: dict[str, Any]) -> dict[str, Any] | None:
        for name in (keys.RTC.CHUNK_OFFSET_SEC, keys.RTC.OBSERVATION_OFFSET_SEC, keys.RTC.ESTIMATED_LATENCY_SEC):
            value = context[name]
            if not _is_finite_number(value) or (name != keys.RTC.OBSERVATION_OFFSET_SEC and value < 0):
                raise ValueError(f'{name} must be finite seconds, non-negative except for the observation offset')
        targets = context[ACTION]
        if not isinstance(targets, list) or any(not isinstance(target, dict) for target in targets):
            raise ValueError('GR00T RTC actions must be a list of dictionaries')
        if not targets:
            return None
        if self._previous_poses is None:
            raise ValueError('RTC overlap requires a preceding prediction in this session')
        horizon = len(self._policy.get_modality_config()[ACTION].delta_indices)
        source_position = context[keys.RTC.CHUNK_OFFSET_SEC] * self._fps
        start = round(source_position)
        if not math.isclose(source_position, start, abs_tol=1e-6) or start + len(targets) > horizon:
            raise ValueError('GR00T prefix must align with the preceding prediction and fit its horizon')
        if context[keys.RTC.OBSERVATION_OFFSET_SEC] > 0:
            raise ValueError('GR00T prefix must cover the observation moment')
        skipped = math.floor(round(-context[keys.RTC.OBSERVATION_OFFSET_SEC] * 1e9) * self._fps / 1e9)
        available = len(targets) - skipped
        if available <= 0:
            return None
        assert self._rtc is not None
        frozen = min(available, math.ceil(round(context[keys.RTC.ESTIMATED_LATENCY_SEC] * 1e9) * self._fps / 1e9))
        overlap = min(available, frozen + math.ceil(self._rtc[keys.RTC.BLEND_SEC] * self._fps))
        actions = {EE_POSE: self._previous_poses[start + skipped : start + skipped + overlap].copy()}
        targets = targets[skipped : skipped + overlap]
        if any(set(target) != {JOINT_POSITION, GRIP} for target in targets):
            raise ValueError('GR00T RTC targets must contain joint_position and gripper_position')
        for name in (JOINT_POSITION, GRIP):
            values = np.asarray([target[name] for target in targets], dtype=np.float32)
            if values.shape != (overlap, STATE_DIMS[name]) or not np.isfinite(values).all():
                raise ValueError(f'Invalid RTC target for {name}')
            actions[name] = values
        # Checkpoint normalization statistics have a full time axis, before processor padding.
        actions = {
            name: np.pad(values, ((0, horizon - len(values)), (0, 0)), mode='edge') for name, values in actions.items()
        }
        return {
            ACTION: actions,
            keys.RTC.ACTION_HORIZON: overlap,
            keys.RTC.OVERLAP_STEPS: overlap,
            keys.RTC.FROZEN_STEPS: frozen,
            keys.RTC.NATIVE_RAMP_RATE: self._rtc[keys.RTC.RAMP_RATE],
        }

    def infer(self, observation: Any) -> Any:
        if self._rtc is None:
            if keys.RTC.FIELD in observation:
                raise ValueError('This session does not enable RTC')
            return self._policy.get_action(observation)
        options = self._rtc_options(observation[keys.RTC.FIELD])
        inputs = {name: value for name, value in observation.items() if name != keys.RTC.FIELD}
        result = self._policy.get_action(inputs, options=options)
        self._previous_poses = result[0][EE_POSE][0].copy()
        return result

    def close(self) -> None:
        self._previous_poses = None


class Gr00tModel(Model):
    """Shared native weights with a fixed observation layout and independently configured sessions."""

    def __init__(
        self,
        policy: Any,
        settings: dict[str, Any],
        metadata: dict[str, Any],
        *,
        fps: float | None = None,
        horizon_sec: float | None = 1.0,
        compress_images: bool = True,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        rtc: dict[str, float] | None = None,
    ):
        self._policy = policy
        self._settings = json.loads(json.dumps(settings, allow_nan=False))
        action_horizon = self._validated_action_horizon()
        super().__init__(
            parameters={
                keys.FPS: settings[model_settings.ACTION_FPS] if fps is None else fps,
                keys.HORIZON_SEC: horizon_sec,
                model_settings.IMAGE_MAPPINGS: settings[model_settings.IMAGE_MAPPINGS],
                keys.COMPRESS_IMAGES: compress_images,
                keys.JPEG_QUALITY: jpeg_quality,
                keys.RTC.FIELD: rtc,
            },
            metadata={**metadata, keys.TYPE: 'groot', keys.EMBODIMENT: EMBODIMENT},
        )
        self.prepare_session(self.parameters)
        width, height = settings[model_settings.IMAGE_SIZE]
        state = {name: np.zeros((1, 1, size), dtype=np.float32) for name, size in STATE_DIMS.items()}
        state[EE_POSE][..., 3:] = [1, 0, 0, 0, 1, 0]
        observation = {
            VIDEO: {
                name: np.zeros((1, 1, height, width, 3), dtype=np.uint8)
                for name in settings[model_settings.IMAGE_MAPPINGS]
            },
            STATE: state,
            LANGUAGE: {TASK: [['pick up the object']]},
        }
        logger.info('Warming GR00T before binding listeners')
        actions, _ = policy.get_action(observation)
        for name in (JOINT_POSITION, GRIP):
            expected = (1, action_horizon, STATE_DIMS[name])
            if actions[name].shape != expected:
                raise ValueError(f'Checkpoint action {name} has shape {actions[name].shape}, expected {expected}')
        if rtc is not None:
            warm = Gr00tSession(policy, {}, rtc, fps=self.parameters[keys.FPS])
            warm.infer({
                **observation,
                keys.RTC.FIELD: {
                    ACTION: [],
                    keys.RTC.CHUNK_OFFSET_SEC: 0.0,
                    keys.RTC.OBSERVATION_OFFSET_SEC: 0.0,
                    keys.RTC.ESTIMATED_LATENCY_SEC: 0.0,
                },
            })
            targets = [
                {name: actions[name][0, index] for name in (JOINT_POSITION, GRIP)} for index in range(action_horizon)
            ]
            warm.infer({
                **observation,
                keys.RTC.FIELD: {
                    ACTION: targets,
                    keys.RTC.CHUNK_OFFSET_SEC: 0.0,
                    keys.RTC.OBSERVATION_OFFSET_SEC: 0.0,
                    keys.RTC.ESTIMATED_LATENCY_SEC: 1 / self.parameters[keys.FPS],
                },
            })
            warm.close()
        policy.reset()
        logger.info('GR00T model is warm')

    def _validated_action_horizon(self) -> int:
        modalities = self._policy.get_modality_config()
        for name, expected in ((VIDEO, set(self._settings[model_settings.IMAGE_MAPPINGS])), (STATE, set(STATE_DIMS))):
            config = modalities[name]
            if set(config.modality_keys) != expected or config.delta_indices != [0]:
                raise ValueError(f'Checkpoint {name} must use current-frame fields {sorted(expected)}; got {config}')
        language = modalities[LANGUAGE]
        if language.modality_keys != [TASK] or language.delta_indices != [0]:
            raise ValueError(f'Checkpoint language must use current-frame {TASK}; got {language}')
        action = modalities[ACTION]
        action_horizon = len(action.delta_indices)
        if (
            not {JOINT_POSITION, GRIP}.issubset(action.modality_keys)
            or not action_horizon
            or action.delta_indices != list(range(action_horizon))
        ):
            raise ValueError(f'Checkpoint must predict consecutive joint and gripper targets; got {action}')
        return action_horizon

    def prepare_session(self, params: dict[str, Any]) -> Gr00tSession:
        rtc = params[keys.RTC.FIELD]
        if rtc is not None:
            if self.parameters[keys.RTC.FIELD] is None:
                raise ValueError('Launch droid_rtc to enable warmed RTC sessions')
            expected = {keys.RTC.CALL_AFTER_SEC, keys.RTC.BLEND_SEC, keys.RTC.MAX_DELAY_SEC, keys.RTC.RAMP_RATE}
            if not isinstance(rtc, dict) or set(rtc) != expected:
                raise ValueError(f'rtc must be None or contain {sorted(expected)}')
            for name, value in rtc.items():
                if not _is_finite_number(value) or value <= 0:
                    raise ValueError(f'rtc.{name} must be positive and finite')
        fps, horizon = params[keys.FPS], params[keys.HORIZON_SEC]
        if rtc is not None and horizon is not None:
            raise ValueError('RTC uses call_after_sec; set horizon_sec=None')
        for name, value in ((keys.FPS, fps), (keys.HORIZON_SEC, horizon)):
            if name == keys.HORIZON_SEC and value is None:
                continue
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        cameras = params[model_settings.IMAGE_MAPPINGS]
        if not isinstance(cameras, dict) or cameras.keys() != self._settings[model_settings.IMAGE_MAPPINGS].keys():
            raise ValueError('image_mappings must retain the checkpoint camera names')
        if any(not isinstance(source, str) or not source for source in cameras.values()):
            raise ValueError('Camera source names must be nonempty strings')
        quality = params[keys.JPEG_QUALITY]
        if type(quality) is not int or not 0 <= quality <= 100:
            raise ValueError('jpeg_quality must be an integer between 0 and 100')
        if type(params[keys.COMPRESS_IMAGES]) is not bool:
            raise ValueError('compress_images must be a boolean')
        return Gr00tSession(self._policy, recipe.client_stack(self._settings, **params), rtc, fps=fps)

    def close(self) -> None:
        del self._policy


def load_model(
    model_source: str,
    settings: dict[str, Any],
    *,
    checkpoint: int | None = None,
    revision: str | None = None,
    device: str = 'cuda',
    fps: float | None = None,
    horizon_sec: float | None = 1.0,
    compress_images: bool = True,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    rtc: dict[str, float] | None = None,
) -> Gr00tModel:
    path, metadata = resolve_checkpoint(model_source, checkpoint, revision)
    model_settings.validate(settings, path)
    # CLI inspection and checkpoint validation do not load the CUDA runtime.
    from .native import Policy  # noqa: PLC0415

    logger.info('Loading GR00T checkpoint %s on %s', path, device)
    policy = Policy(embodiment_tag=EMBODIMENT, model_path=str(path), device=device, strict=True)
    return Gr00tModel(
        policy,
        settings,
        metadata,
        fps=fps,
        horizon_sec=horizon_sec,
        compress_images=compress_images,
        jpeg_quality=jpeg_quality,
        rtc=rtc,
    )


websocket = cfn.Config(WebsocketWire, served_address=cfn.Config(ServedHostPort, host='0.0.0.0', port=8000))


@cfn.config(served_address={'host': '0.0.0.0', 'port': 8001})
def grpc(served_address: dict[str, Any]):
    # gRPC is an independently installed transport extra.
    from positronic_model_server.grpc_wire import GrpcWire  # noqa: PLC0415

    return GrpcWire(ServedHostPort(**served_address))


@cfn.config(
    model={'model_source': HF_MODEL_PREFIX + BASE_MODEL, 'checkpoint': None, 'revision': None, 'device': 'cuda'},
    pipeline={
        'settings': model_settings.droid(),
        keys.FPS: None,
        keys.HORIZON_SEC: 1.0,
        keys.COMPRESS_IMAGES: True,
        keys.JPEG_QUALITY: DEFAULT_JPEG_QUALITY,
        keys.RTC.FIELD: None,
    },
    websocket=websocket,
    grpc=None,
    idle_timeout_min=None,
)
def serve(model: dict[str, Any], pipeline: dict[str, Any], websocket, grpc, idle_timeout_min: float | None):
    """Launch one checkpoint; model construction and warm-up run on the model worker thread."""
    if model['checkpoint'] is not None:
        model = {**model, 'checkpoint': int(model['checkpoint'])}
    server = ModelServer(
        partial(load_model, **model, **pipeline),
        idle_timeout_min=idle_timeout_min,
        auth_token=os.environ.get(AUTH_TOKEN_ENV),
    )
    server.serve([transport for transport in (websocket, grpc) if transport is not None])


def main() -> None:
    droid = serve.override(**{'model.model_source': os.environ.get(MODEL_SOURCE_ENV, HF_MODEL_PREFIX + BASE_MODEL)})
    three_cameras = droid.override(**{'pipeline.settings': model_settings.droid_three_cameras()})
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    rtc = droid.override(**{
        'pipeline.rtc': {
            keys.RTC.CALL_AFTER_SEC: 0.5,
            keys.RTC.BLEND_SEC: 0.5,
            keys.RTC.MAX_DELAY_SEC: 1.0,
            keys.RTC.RAMP_RATE: 5.0,
        },
        'pipeline.horizon_sec': None,
    })
    cfn.cli({'serve': droid, 'droid': droid, 'droid_three_cameras': three_cameras, 'droid_rtc': rtc})


if __name__ == '__main__':
    main()
