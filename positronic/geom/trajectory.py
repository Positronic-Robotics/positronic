from positronic import geom


class AbsoluteTrajectory(list):
    def __str__(self):
        return "AbsoluteTrajectory" + super().__str__()

    def __repr__(self):
        return "AbsoluteTrajectory" + super().__repr__()

    def to_relative(self) -> 'RelativeTrajectory':
        relative_positions = RelativeTrajectory()

        for i in range(1, len(self)):
            relative_positions.append(self[i-1].inv * self[i])

        return relative_positions


class RelativeTrajectory(list):
    def __str__(self):
        return "RelativeTrajectory" + super().__str__()

    def __repr__(self):
        return "RelativeTrajectory" + super().__repr__()

    def to_absolute(self, start_position: geom.Transform3D) -> AbsoluteTrajectory:
        absolute_positions = AbsoluteTrajectory([start_position])

        for pos in self:
            absolute_positions.append(absolute_positions[-1] * pos)

        return absolute_positions
