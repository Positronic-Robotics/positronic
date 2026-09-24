"""Model loading and policy deployment configuration for the inference server."""

from abc import ABC, abstractmethod
from collections.abc import Callable
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
        return {}

    def close(self) -> None:
        """Release the model's resources after all calls have finished."""
        return None


class ModelSource(ABC):
    """The checkpoint a server loads at launch: cheap to build and to compare, and loaded once by ``load``."""

    @abstractmethod
    def checkpoint_id(self) -> str:
        """The id of the checkpoint this source serves. The server reads it once, at launch."""

    @abstractmethod
    def load(self, checkpoint_id: str, on_progress: Callable[[str], None] | None = None) -> Model: ...

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__


@dataclass
class PolicyDeployment:
    """A model source, a client stack of processors and codecs, and an optional server codec."""

    source: ModelSource
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
