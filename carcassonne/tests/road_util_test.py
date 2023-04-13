import unittest

from ..carcassonne_game_state import CarcassonneGameState
from ..objects.coordinate import Coordinate
from ..objects.coordinate_with_side import CoordinateWithSide
from ..objects.road import Road
from ..objects.side import Side
from ..tile_sets.base_deck import base_tiles
from ..utils.road_util import RoadUtil


class TestRoadUtil(unittest.TestCase):

    def test_find_road(self):
        """
        Find the positions for a city with a top and a bottom
        """

        # Given
        game_state: CarcassonneGameState = CarcassonneGameState()

        crossroads = base_tiles["crossroads"]

        game_state.board = [[None for column in range(1)] for row in range(2)]

        game_state.board[0][0] = crossroads
        game_state.board[1][0] = crossroads

        # When
        road: Road = RoadUtil.find_road(
            game_state=game_state,
            road_position=CoordinateWithSide(Coordinate(0, 0), Side.BOTTOM)
        )

        # Then
        self.assertTrue(road.finished)
        self.assertEqual(2, len(road.road_positions))
        self.assertIn(CoordinateWithSide(Coordinate(0, 0), Side.BOTTOM), road.road_positions)
        self.assertIn(CoordinateWithSide(Coordinate(1, 0), Side.TOP), road.road_positions)
