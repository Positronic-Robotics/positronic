from typing import List, Optional

import numpy as np

import geom


class AbsoluteTrajectory:
    def __init__(self, absolute_positions: List[geom.Transform3D]):
        self.absolute_positions = absolute_positions

    def __len__(self):
        return len(self.absolute_positions)

    def __iter__(self):
        return iter(self.absolute_positions)

    def __getitem__(self, index: int) -> geom.Transform3D:
        return self.absolute_positions[index]

    def to_relative(self) -> 'RelativeTrajectory':
        relative_positions = []

        for i in range(1, len(self.absolute_positions)):
            relative_positions.append(self.absolute_positions[i-1].inv * self.absolute_positions[i])

        return RelativeTrajectory(relative_positions)


class RelativeTrajectory:
    def __init__(self, relative_positions: List[geom.Transform3D]):
        self.relative_positions = relative_positions

    def __len__(self):
        return len(self.relative_positions)

    def __iter__(self):
        return iter(self.relative_positions)

    def __getitem__(self, index: int) -> geom.Transform3D:
        return self.relative_positions[index]

    def to_absolute(self, start_position: Optional[geom.Transform3D] = None) -> AbsoluteTrajectory:
        if start_position is None:
            start_position = geom.Transform3D()

        absolute_positions = [start_position]

        for pos in self.relative_positions:
            absolute_positions.append(absolute_positions[-1] * pos)

        return AbsoluteTrajectory(absolute_positions)

