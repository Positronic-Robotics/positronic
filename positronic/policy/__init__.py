from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Policy(ABC):
    @abstractmethod
    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        pass

    def reset(self):
        return None
