from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    @abstractmethod
    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        pass

    def reset(self):
        return None

    @property
    def meta(self) -> dict[str, Any]:
        return {}
