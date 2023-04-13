from carcassonne.objects.coordinate import Coordinate
from .carcassonne_game_state import CarcassonneGameState
from carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from carcassonne.tile_sets.tile_sets import TileSet
from .game_state import GameState
from .game import Game
from carcassonne.carcassonne_game import CarcassonneGame as LibCarcassonneGame
from carcassonne.carcassonne_game_state import CarcassonneGameState as LibCarcassonneGameState


class CarcassonneGame(Game[CarcassonneGameState]):
    def __init__(self) -> None:
        super().__init__()

    def get_initial_state(self) -> GameState:
        players = 2
        tile_sets= [TileSet.BASE, TileSet.INNS_AND_CATHEDRALS]
        supplementary_rules = [SupplementaryRule.ABBOTS, SupplementaryRule.FARMERS]
        if TileSet.THE_RIVER in tile_sets:
            raise NotImplementedError("The river has bug related to get_river_rotation_ends (need to find the right starting end for previous_river_ends). Do not use")

        self.supplementary_rules = supplementary_rules
        lib_state = LibCarcassonneGameState(
            tile_sets=tile_sets,
            players=players,
            supplementary_rules=supplementary_rules,
            board_size=(10, 10),
            starting_position=Coordinate(5, 5)
        )
        return CarcassonneGameState(lib_state)
