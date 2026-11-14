"""Serve FLUX 3 Action in its native interpreter through the thin model server."""

import logging
import math
import os
from collections import deque
from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from positronic_model_server.serialization import check_int_range
from positronic_model_server.server import Model, ModelServer, Session
from positronic_model_server.server_wire import ServedHostPort
from positronic_model_server.websocket_wire import WebsocketWire

from . import keys, recipe

logger = logging.getLogger(__name__)


class Flux3Session(Session):
    """Use recent round-trip durations to freeze the part of the plan executed during inference."""

    def __init__(self, policy: Any, client_stack: dict, *, fps: float, rtc: dict[str, float] | None):
        super().__init__(client_stack)
        self._policy = policy
        self._fps = fps
        self._rtc = rtc
        self._latencies: deque[float] = deque(maxlen=5)
        self._queries = 0

    def _committed_actions(self, context: dict) -> np.ndarray | None:
        for timing in context[keys.CALL_TIMINGS]:
            elapsed = timing[keys.ELAPSED_SEC]
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('elapsed_sec must be finite and nonnegative')
            self._latencies.append(elapsed)
        assert self._rtc is not None
        delay = max(self._latencies, default=0.0)
        if delay > self._rtc[keys.RTC.MAX_DELAY_SEC]:
            raise ValueError('Observed inference latency exceeds rtc.max_delay_sec')
        prefix = context[keys.RTC.ACTION_PREFIX]
        for name in (keys.RTC.CHUNK_OFFSET_SEC, keys.RTC.OBSERVATION_OFFSET_SEC):
            value = prefix[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite seconds')
        offset = prefix[keys.RTC.CHUNK_OFFSET_SEC] * self._fps
        if offset < 0 or not math.isclose(offset, round(offset), abs_tol=1e-6):
            raise ValueError('The prefix must align with the preceding action grid')
        targets = prefix[keys.RTC.PREFIX_ACTIONS]
        if not isinstance(targets, list) or any(not isinstance(t, dict) or set(t) != {keys.ACTION} for t in targets):
            raise ValueError('RTC targets must contain action vectors')
        if not targets:
            return None
        if not self._queries or round(offset) + len(targets) > self._policy.config.chunk_size:
            raise ValueError('RTC prefix must fit a preceding prediction in this session')
        observation_offset = prefix[keys.RTC.OBSERVATION_OFFSET_SEC]
        if observation_offset > 0:
            raise ValueError('RTC prefix must cover the observation moment')
        actions = np.asarray([target[keys.ACTION] for target in targets], dtype=np.float32)
        if actions.shape != (len(targets), 8) or not np.isfinite(actions).all():
            raise ValueError('RTC actions must be finite eight-dimensional absolute commands')
        skipped = math.floor(round(-observation_offset * 1e9) * self._fps / 1e9)
        count = math.ceil(round(delay * 1e9) * self._fps / 1e9)
        committed = actions[skipped : skipped + count].copy()
        return committed if len(committed) else None

    def infer(self, observation: Any) -> Any:
        if self._rtc is None:
            if keys.RTC.FIELD in observation:
                raise ValueError('This session does not enable RTC')
            committed = None
        else:
            committed = self._committed_actions(observation[keys.RTC.FIELD])
        result = self._policy.predict(observation, committed, seed=self._queries + 1)
        self._queries += 1
        return {keys.ACTION: result}


class Flux3Model(Model):
    """Shared stateless DROID weights; each session owns its latency history and random sequence."""

    def __init__(self, policy: Any, *, metadata: dict, image_mappings: dict[str, str], rtc: dict[str, float] | None):
        self._policy = policy
        super().__init__(
            parameters={
                'horizon_sec': None,
                'image_mappings': image_mappings,
                'compress_images': True,
                'jpeg_quality': 90,
                keys.RTC.FIELD: rtc,
            },
            metadata={**metadata, keys.ACTION_FPS: policy.config.fps},
        )
        self.prepare_session(self.parameters)

    def _validate_rtc(self, rtc: dict, horizon: float | None) -> None:
        if self.parameters[keys.RTC.FIELD] is None:
            raise ValueError('Launch droid_rtc or droid_3cam_rtc to enable warmed RTC sessions')
        cfg = self._policy.config
        if cfg.rtc_max_prefix <= 0 or cfg.quantization is not None or cfg.action_parameterization == 'joint_delta':
            raise ValueError('RTC requires a compatible BF16 checkpoint trained with rtc_max_prefix > 0')
        if not isinstance(rtc, dict) or set(rtc) != {keys.RTC.CALL_AFTER_SEC, keys.RTC.MAX_DELAY_SEC}:
            raise ValueError('rtc requires call_after_sec and max_delay_sec')
        for name, value in rtc.items():
            if type(value) not in (int, float) or not 0 < value < math.inf:
                raise ValueError(f'rtc.{name} must be finite and positive')
        if math.ceil(round(rtc[keys.RTC.MAX_DELAY_SEC] * 1e9) * cfg.fps / 1e9) > cfg.rtc_max_prefix:
            raise ValueError('rtc.max_delay_sec exceeds the checkpoint RTC prefix capacity')
        if rtc[keys.RTC.CALL_AFTER_SEC] >= cfg.chunk_size / cfg.fps:
            raise ValueError('rtc.call_after_sec must leave actions available for overlap')
        if horizon is not None:
            raise ValueError('RTC uses call_after_sec; set horizon_sec=None')

    def prepare_session(self, params: dict[str, Any]) -> Flux3Session:
        horizon = params['horizon_sec']
        if horizon is not None and (type(horizon) not in (int, float) or not 0 < horizon < math.inf):
            raise ValueError('horizon_sec must be finite and positive')
        mappings = params['image_mappings']
        if (
            not isinstance(mappings, dict)
            or set(mappings) != set(keys.CAMERAS)
            or any(not isinstance(source, str) or not source for source in mappings.values())
        ):
            raise ValueError('image_mappings must name all three DROID views')
        check_int_range(params['jpeg_quality'], 0, 100, 'jpeg_quality')
        if type(params['compress_images']) is not bool:
            raise ValueError('compress_images must be a boolean')
        rtc = params[keys.RTC.FIELD]
        if rtc is not None:
            self._validate_rtc(rtc, horizon)
        return Flux3Session(
            self._policy,
            recipe.client_stack(fps=self._policy.config.fps, **params),
            fps=self._policy.config.fps,
            rtc=rtc,
        )

    def close(self) -> None:
        self._policy.close()

    def warmup(self) -> None:
        """Compile ordinary and committed-prefix inference before accepting sessions."""
        logger.info('Warming FLUX 3 Action before binding listeners')
        width, height = keys.VIEW_SIZE
        observation = {name: np.zeros((height, width, 3), np.uint8) for name in keys.CAMERAS}
        observation.update({
            keys.JOINT_POSITION: np.zeros(7),
            keys.GRIPPER_POSITION: np.zeros(1),
            keys.PROMPT: 'Pick up the object.',
        })
        actions = self._policy.predict(observation, None, seed=0)
        if self.parameters[keys.RTC.FIELD] is not None:
            self._policy.predict(observation, actions[: self._policy.config.rtc_max_prefix].copy(), seed=0)


def load_model(
    checkpoint: str,
    revision: str | None,
    subfolder: str | None,
    *,
    image_mappings: dict[str, str],
    rtc: dict[str, float] | None,
    device: str = 'cuda',
    compile_dit: bool = True,
    offload_text_encoder: bool = False,
) -> Flux3Model:
    from .native import Policy  # noqa: PLC0415

    policy = Policy(
        checkpoint,
        revision,
        subfolder,
        device=device,
        compile_dit=compile_dit,
        offload_text_encoder=offload_text_encoder,
        rtc=rtc is not None,
    )
    try:
        model = Flux3Model(
            policy,
            image_mappings=image_mappings,
            rtc=rtc,
            metadata={
                'type': 'flux3_action',
                'checkpoint_id': checkpoint,
                'revision': revision,
                'subfolder': subfolder,
            },
        )
        model.warmup()
        return model
    except BaseException:
        policy.close()
        raise


@cfn.config(
    checkpoint='black-forest-labs/flux-3-action-droid',
    revision=None,
    subfolder=None,
    image_mappings=recipe.droid_cameras(),
    rtc=None,
    device='cuda',
    compile_dit=True,
    offload_text_encoder=False,
    host='0.0.0.0',
    port=8000,
)
def serve(
    checkpoint: str,
    revision: str | None,
    subfolder: str | None,
    image_mappings: dict[str, str],
    rtc: dict[str, float] | None,
    device: str,
    compile_dit: bool,
    offload_text_encoder: bool,
    host: str,
    port: int,
):
    server = ModelServer(
        partial(
            load_model,
            checkpoint,
            revision,
            subfolder,
            image_mappings=image_mappings,
            rtc=rtc,
            device=device,
            compile_dit=compile_dit,
            offload_text_encoder=offload_text_encoder,
        ),
        auth_token=os.environ.get(keys.AUTH_TOKEN_ENV),
    )
    server.serve([WebsocketWire(ServedHostPort(host, port))])


def main() -> None:
    droid = serve.override(
        checkpoint=os.environ.get(keys.MODEL_SOURCE_ENV, 'black-forest-labs/flux-3-action-droid'),
        revision=os.environ.get(keys.MODEL_REVISION_ENV),
        subfolder=os.environ.get(keys.MODEL_SUBFOLDER_ENV),
    )
    three_cameras = droid.override(image_mappings=recipe.droid_cameras(three_cameras=True))
    rtc = {keys.RTC.CALL_AFTER_SEC: 0.5, keys.RTC.MAX_DELAY_SEC: 1.0}
    logging.basicConfig(level=logging.INFO)
    cfn.cli({
        'droid': droid,
        'droid_3cam': three_cameras,
        'droid_rtc': droid.override(rtc=rtc),
        'droid_3cam_rtc': three_cameras.override(rtc=rtc),
    })


if __name__ == '__main__':
    main()
