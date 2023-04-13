from ..objects.coordinate_with_side import CoordinateWithSide


class City:
    def __init__(self, city_positions: [CoordinateWithSide], finished: bool):
        self.city_positions = city_positions
        self.finished = finished

    def __eq__(self, other):
        return set(self.city_positions) == set(other.city_positions) and self.finished == other.finished
    
    def __hash__(self):
        return hash((frozenset(self.city_positions), self.finished))
