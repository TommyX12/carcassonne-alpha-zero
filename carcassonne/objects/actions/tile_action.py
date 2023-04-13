from ..actions.action import Action
from ..coordinate import Coordinate
from ..tile import Tile


class TileAction(Action):
    def __init__(self, tile: Tile, coordinate: Coordinate, tile_rotations: int):
        super().__init__()
        self.tile = tile
        self.coordinate = coordinate
        self.tile_rotations = tile_rotations

    def __str__(self):
        return f"TileAction({self.tile.description}, {self.coordinate}, {self.tile_rotations})"

    def __repr__(self):
        return str(self)

    def to_json(self, tile_registry):
        return [tile_registry.tile_to_json(self.tile), self.coordinate.to_json(), self.tile_rotations]

    @staticmethod
    def from_json(data, tile_registry):
        return TileAction(tile_registry.tile_from_json(data[0]), Coordinate.from_json(data[1]), data[2])
