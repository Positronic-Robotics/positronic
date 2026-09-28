"""Servers a policy runs against, and the session an episode holds on one."""

import collections.abc as cabc
from abc import ABC, abstractmethod
from contextlib import closing
from typing import Any

import numpy as np
from positronic_model_server.protocol import ProtocolVersion
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, encode_jpeg
from positronic_wire import registry
from positronic_wire.wire import SessionAddress

from positronic import telemetry, telemetry_keys
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.policy import keys as policy_keys
from positronic.utils import flatten_dict

from .base import Obs, Policy, Processor
from .compatibility import from_v1_spec
from .spec import from_spec


class Session(ABC):
    """One episode's connection to a server: the stack the server declares for the rig, and one inference per call.

    The runtime opens it at episode start and makes one call at a time. It closes the session after the last call.
    """

    local_stack: Policy

    @abstractmethod
    def __call__(self, obs: Obs) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...


class Server(ABC):
    """An address and the rules of the wire that reaches it. It holds no connection."""

    @abstractmethod
    def open(self) -> Session: ...

    def meta(self) -> dict[str, Any]:
        """Model and configuration metadata shared across episodes."""
        return {}


def _prepare_value(value: Any, jpeg_quality: int) -> Any:
    # Codecs nest images inside dicts and lists (e.g. GR00T), so recurse to reach every image array.
    if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
        # A raw HD frame — especially a (T, H, W, 3) stack — can exceed a proxy's message cap.
        return encode_jpeg(value, jpeg_quality)
    if isinstance(value, cabc.Mapping):
        return {k: _prepare_value(v, jpeg_quality) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_prepare_value(v, jpeg_quality) for v in value)
    return value


def prepare_obs(
    obs: cabc.Mapping[str, Any], compress_images: bool, jpeg_quality: int = DEFAULT_JPEG_QUALITY
) -> dict[str, Any]:
    if not compress_images:
        return dict(obs)
    return {key: _prepare_value(value, jpeg_quality) for key, value in obs.items()}


def round_trip(
    session: InferenceSession,
    obs: cabc.Mapping[str, Any],
    compress_images: bool,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[dict[str, Any]] | dict[str, Any]:
    """One inference over the wire, timed as the ``policy.infer`` span.

    Image preparation has its own span; the inference span covers only the server round trip.
    """
    with telemetry.span(telemetry_keys.SPAN_POLICY_PREPARE):
        prepared = prepare_obs(obs, compress_images, jpeg_quality)
    with telemetry.span(telemetry_keys.SPAN_POLICY_INFER) as span:
        try:
            return session.infer(prepared)
        finally:
            served = {f'{telemetry_keys.ATTR_SERVED_PREFIX}{k}': v for k, v in session.served_timing.items()}
            telemetry.set_attrs(span, **served)


def declared_stack(meta: cabc.Mapping[str, Any], protocol_version: ProtocolVersion) -> Processor:
    """The client stack a server's handshake declares, in the form ``protocol_version`` runs."""
    declared = meta.get(offboard_keys.LOCAL_STACK)
    if declared is None:
        raise ValueError('Server declares no client processor stack')
    if protocol_version is ProtocolVersion.V1:
        stack = from_v1_spec(declared, meta)
    else:
        stack = from_spec(declared)
    if not isinstance(stack, Processor):
        raise ValueError('The declared client stack must be a processor')
    return stack


class WireSession(Session):
    """A session on a positronic wire server. Images go as JPEG when the handshake asks for compressed images."""

    def __init__(self, session: InferenceSession, jpeg_quality: int) -> None:
        self.local_stack = declared_stack(session.metadata, session.protocol_version)
        self._session = session
        self._compress_images = bool(session.metadata.get(offboard_keys.COMPRESS_IMAGES))
        self._jpeg_quality = jpeg_quality

    def __call__(self, obs: Obs) -> list[dict[str, Any]] | dict[str, Any]:
        return round_trip(self._session, obs, self._compress_images, self._jpeg_quality)

    def close(self) -> None:
        self._session.close()


class WireServer(Server):
    """A server that speaks the positronic wire protocol.

    ``wire`` names the transport and ``address`` is the address it dials. ``jpeg_quality`` sets the JPEG quality of
    images sent to a server that asks for compressed images.
    """

    def __init__(
        self,
        wire: str,
        address: SessionAddress,
        *,
        headers: dict[str, str] | None = None,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        self._client = InferenceClient(
            registry.client_wire(wire), address, headers=headers, infer_timeout=infer_timeout
        )
        self._jpeg_quality = jpeg_quality
        # The handshake metadata of the last session, which `meta` reports.
        self._served: dict[str, Any] | None = None

    def open(self) -> WireSession:
        session = self._client.new_session()
        try:
            self._served = dict(session.metadata)
            return WireSession(session, self._jpeg_quality)
        except BaseException:
            session.close()
            raise

    def meta(self) -> dict[str, Any]:
        if self._served is None:
            with closing(self._client.new_session()) as session:
                self._served = dict(session.metadata)
        meta: dict[str, Any] = {policy_keys.TYPE: 'remote', policy_keys.SERVER: self._served}
        if self._served.get(offboard_keys.COMPRESS_IMAGES):
            meta[policy_keys.JPEG_QUALITY] = self._jpeg_quality
        return flatten_dict(meta)
