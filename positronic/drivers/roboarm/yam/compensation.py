from collections.abc import Sequence
from dataclasses import dataclass


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
    # Two joint speeds in rad/s: no friction below the first, all of it from the second, linear between. A joint at rest
    # reads speeds of a few mrad/s, and the sign of that noise flips the friction. None adds it at any speed.
    coulomb_friction_fade_rad_s: Sequence[float] | None = None

    def __post_init__(self) -> None:
        # Keep tuple copies: a list that the caller changes later must not reach the chain
        if self.gravity_comp_factor is not None:
            object.__setattr__(self, 'gravity_comp_factor', tuple(self.gravity_comp_factor))
        if self.coulomb_friction is not None:
            object.__setattr__(self, 'coulomb_friction', tuple(self.coulomb_friction))
        if self.coulomb_friction_fade_rad_s is not None:
            object.__setattr__(self, 'coulomb_friction_fade_rad_s', tuple(self.coulomb_friction_fade_rad_s))


I2RT_COMPENSATION = Compensation()  # what i2rt adds when it is asked for nothing
