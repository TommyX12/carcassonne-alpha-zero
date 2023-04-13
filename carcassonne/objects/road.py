from ..objects.coordinate_with_side import CoordinateWithSide


class Road:
    def __init__(self, road_positions: [CoordinateWithSide], finished: bool):
        self.road_positions = road_positions
        self.finished = finished

    def __eq__(self, other):
        return set(self.road_positions) == set(other.road_positions) and self.finished == other.finished

    def __hash__(self):
        return hash((frozenset(self.road_positions), self.finished))
