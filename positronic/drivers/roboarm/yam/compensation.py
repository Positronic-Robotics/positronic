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


I2RT_COMPENSATION = Compensation()  # what i2rt adds when it is asked for nothing
