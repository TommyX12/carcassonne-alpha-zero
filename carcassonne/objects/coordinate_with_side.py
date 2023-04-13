from ..objects.coordinate import Coordinate
from ..objects.side import Side


class CoordinateWithSide:

    def __init__(self, coordinate: Coordinate, side: int):
        self.coordinate = coordinate
        self.side = side

    def __eq__(self, other):
        return self.coordinate == other.coordinate and self.side == other.side

    def __hash__(self):
        return hash((self.coordinate, self.side))

    def __str__(self):
        return f"{self.coordinate} {self.side}"

    def __repr__(self) -> str:
        return str(self)

    def to_json(self):
        return [self.coordinate.to_json(), self.side]
    
    @staticmethod
    def from_json(data: list):
        return CoordinateWithSide(Coordinate.from_json(data[0]), data[1])
