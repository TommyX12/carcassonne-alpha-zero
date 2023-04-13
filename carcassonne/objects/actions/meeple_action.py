from ..actions.action import Action
from ..coordinate_with_side import CoordinateWithSide
from ..meeple_type import MeepleType


class MeepleAction(Action):
    def __init__(self, meeple_type: int, coordinate_with_side: CoordinateWithSide, remove: bool = False):
        super().__init__()
        self.meeple_type = meeple_type
        self.coordinate_with_side = coordinate_with_side
        self.remove = remove

    def __str__(self):
        return f"MeepleAction({self.meeple_type}, {self.coordinate_with_side}, {self.remove})"

    def __repr__(self):
        return str(self)

    def to_json(self):
        return [self.meeple_type, self.coordinate_with_side.to_json(), self.remove]

    @staticmethod
    def from_json(data):
        return MeepleAction(data[0], CoordinateWithSide.from_json(data[1]), data[2])
