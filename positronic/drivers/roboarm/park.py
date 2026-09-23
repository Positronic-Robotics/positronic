from dataclasses import dataclass


@dataclass(frozen=True)
class ParkTuning:
    """How close a park must land, and what it may spend to get there.

    A position servo holds its chain a steady distance short of the reference, and that distance differs
    between two arms. The park measures it and asks for a reference past the parking pose, by up to
    ``max_correction_rad``. The defaults fit the arm the driver was brought up on.
    """

    tolerance_rad: float = 0.005  # every joint must rest this close to the parking pose
    attempts: int = 6  # correction passes before the park fails
    max_correction_rad: float = 0.05  # the most the reference may lie past the parking pose, per joint
