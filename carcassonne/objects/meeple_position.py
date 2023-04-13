from ..objects.coordinate_with_side import CoordinateWithSide
from ..objects.meeple_type import MeepleType


class MeeplePosition:
    def __init__(self, meeple_type: int, coordinate_with_side: CoordinateWithSide):
        self.meeple_type = meeple_type
        self.coordinate_with_side = coordinate_with_side

    def __eq__(self, other):
        return self.meeple_type == other.meeple_type and self.coordinate_with_side == other.coordinate_with_side

    def __hash__(self):
        return hash((self.meeple_type, self.coordinate_with_side))

    def to_json(self):
        return [self.meeple_type, self.coordinate_with_side.to_json()]

    @staticmethod
    def from_json(data: list):
        return MeeplePosition(data[0], CoordinateWithSide.from_json(data[1]))
