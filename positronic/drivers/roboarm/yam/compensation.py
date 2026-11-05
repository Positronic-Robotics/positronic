import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrictionFade:
    """How the Coulomb friction of a joint fades at low speed: none below ``zero_below_rad_s``, all of it from
    ``full_from_rad_s``, linear between. A joint at rest reads speeds of a few mrad/s, and the sign of that noise
    flips a friction that does not fade."""

    zero_below_rad_s: float
    full_from_rad_s: float

    def __post_init__(self) -> None:
        low, full = self.zero_below_rad_s, self.full_from_rad_s
        if not (math.isfinite(low) and math.isfinite(full) and 0 <= low < full):
            raise ValueError(f'a friction fade needs finite speeds, 0 <= zero_below_rad_s < full_from_rad_s: {self}')

    def share(self, speed: np.ndarray) -> np.ndarray:
        """The part of its Coulomb friction that a joint gets at ``speed``."""
        ramp = (np.abs(speed) - self.zero_below_rad_s) / (self.full_from_rad_s - self.zero_below_rad_s)
        return np.clip(ramp, 0.0, 1.0)


@dataclass(frozen=True)
class Compensation:
    """The torque that i2rt adds to the PD of one YAM chain: the gravity of its model, and Coulomb friction.

    A field left None keeps i2rt's own value. A joint that rests a steady offset below its target is under-
    compensated, and the offset is the torque it lacks divided by its position gain.
    """

    gravity_comp_factor: Sequence[float] | None = None  # per arm joint, scales the gravity torque of the model
    ee_mass: float | None = None  # kg; replaces the mass of the gripper body in the model
    use_coulomb_friction: bool = False
    coulomb_friction: Sequence[float] | None = None  # Nm per arm joint, in the direction that the joint moves
    coulomb_friction_fade: FrictionFade | None = None  # None adds the friction at any speed

    def __post_init__(self) -> None:
        # Keep tuple copies: a list that the caller changes later must not reach the chain
        if self.gravity_comp_factor is not None:
            object.__setattr__(self, 'gravity_comp_factor', tuple(self.gravity_comp_factor))
        if self.coulomb_friction is not None:
            object.__setattr__(self, 'coulomb_friction', tuple(self.coulomb_friction))


I2RT_COMPENSATION = Compensation()  # what i2rt adds when it is asked for nothing
