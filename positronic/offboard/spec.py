"""The model an inference server loads, and the policy deployment it serves the model through."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.policy.base import Obs, Policy
from positronic.policy.codec import Codec


class Model(ABC):
    """A loaded inference callable and the resources it owns."""

    @abstractmethod
    def __call__(self, obs: Obs, *, session_id: str) -> Any: ...

    def end_session(self, session_id: str) -> None:
        """Release one session's state after its calls finish; keep the loaded model available."""
        return None

    def meta(self) -> dict[str, Any]:
        """What each session's handshake reports about the model, its ``offboard.keys.CHECKPOINT_ID`` among it."""
        return {}

    def close(self) -> None:
        """Release the model's resources after all calls have finished."""
        return None


@dataclass
class PolicyDeployment:
    """A client stack of processors and codecs, and an optional server codec, which a session may retune."""

    local: Policy
    codec: Codec | None = None
    compress_images: bool = False

    def __post_init__(self) -> None:
        if (
            self.codec is not None
            and roboarm_keys.EE_FRAME in self.codec.meta
            and roboarm_keys.EE_FRAME in self.local.meta()
        ):
            raise ValueError('Only one side of a deployment may convert the end-effector frame')
