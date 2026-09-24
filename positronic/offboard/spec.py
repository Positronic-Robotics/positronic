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
    """Configuration that discovers and loads models; loaded resources belong to the returned model."""

    @abstractmethod
    def get_models(self) -> list[str]:
        """Available IDs, oldest first. The default resolver selects the last entry."""

    def resolve(self, model_id: str | None) -> str:
        models = self.get_models()
        if model_id is None:
            return models[-1]
        if model_id not in models:
            raise ValueError(f'Unknown model {model_id!r}. Available: {models}')
        return model_id

    @abstractmethod
    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Model: ...

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__


@dataclass
class PolicyDeployment:
    """A model source, a client stack of processors and codecs, and an optional server codec."""

    source: ModelSource
    local: Policy
    codec: Codec | None = None
    compress_images: bool = True

    def __post_init__(self) -> None:
        if (
            self.codec is not None
            and roboarm_keys.EE_FRAME in self.codec.meta
            and roboarm_keys.EE_FRAME in self.local.meta()
        ):
            raise ValueError('Only one side of a deployment may convert the end-effector frame')
