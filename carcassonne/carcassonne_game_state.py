import random
from typing import Any, Dict, List, Optional

from .objects.actions.action import Action
from .objects.actions.pass_action import PassAction
from .objects.actions.meeple_action import MeepleAction
from .objects.meeple_position import MeeplePosition
from .objects.actions.tile_action import TileAction
from .objects.coordinate import Coordinate
from .objects.game_phase import GamePhase
from .objects.rotation import Rotation
from .objects.tile import Tile
from .tile_sets.base_deck import base_tile_counts, base_tiles
from .tile_sets.inns_and_cathedrals_deck import inns_and_cathedrals_tiles, \
    inns_and_cathedrals_tile_counts
from .tile_sets.supplementary_rules import SupplementaryRule
from .tile_sets.the_river_deck import the_river_tiles, the_river_tile_counts
from .tile_sets.tile_sets import TileSet
from utils.debug import debug


class TileRegistry(object):
    def __init__(self):
        self.tile_dict: Dict[str, Tile] = {}

    def add_deck(self, deck: Dict[str, Tile]):
        for tile in deck.values():
            if tile.description in self.tile_dict:
                raise ValueError(f"Tile {tile.description} already exists in registry")

            self.tile_dict[tile.description] = tile

    def tile_to_json(self, tile: Tile):
        return {"description": tile.description, "turns": tile.turns}

    def tile_from_json(self, data: dict):
        id: str = data["description"]
        turns: int = data["turns"]
        return self.tile_dict[id].turn(turns)


tile_registry = TileRegistry()
tile_registry.add_deck(base_tiles)
tile_registry.add_deck(inns_and_cathedrals_tiles)


def action_from_json(data: dict):
    if data["type"] == "TileAction":
        return TileAction.from_json(data["data"], tile_registry)

    if data["type"] == "MeepleAction":
        return MeepleAction.from_json(data["data"])

    if data["type"] == "PassAction":
        return PassAction()


def action_to_json(action: Action):
    if isinstance(action, TileAction):
        return {"type": "TileAction", "data": action.to_json(tile_registry)}

    if isinstance(action, MeepleAction):
        return {"type": "MeepleAction", "data": action.to_json()}

    if isinstance(action, PassAction):
        return {"type": "PassAction"}


class CarcassonneGameState:

    def __init__(
            self,
            tile_sets: [TileSet] = (TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS),
            supplementary_rules: [SupplementaryRule] = (SupplementaryRule.FARMERS, SupplementaryRule.ABBOTS),
            players: int = 2,
            board_size: (int, int) = (35, 35),
            starting_position: Coordinate = Coordinate(6, 15),
            empty=False,
    ):
        if not empty:
            self.deck = self.initialize_deck(tile_sets=tile_sets)
            self.supplementary_rules: [SupplementaryRule] = supplementary_rules
            self.board: [[Tile]] = [[None for column in range(board_size[1])] for row in range(board_size[0])]
            self.starting_position: Coordinate = starting_position
            self.next_tile: Optional[Tile] = self.deck.pop(0)
            self.players = players
            self.meeples = [7 for _ in range(players)]
            self.abbots = [1 if SupplementaryRule.ABBOTS in supplementary_rules else 0 for _ in range(players)]
            self.big_meeples = [1 if TileSet.INNS_AND_CATHEDRALS in tile_sets else 0 for _ in range(players)]
            self.placed_meeples: List[List[MeeplePosition]] = [[] for _ in range(players)]
            self.scores: [int] = [0 for _ in range(players)]
            self.current_player = 0
            self.phase = GamePhase.TILES
            self.last_tile_action: Optional[TileAction] = None
            self.last_river_rotation: Rotation = Rotation.NONE

    def to_json(self):
        data = {
            "deck": [tile_registry.tile_to_json(tile) for tile in self.deck],
            "supplementary_rules": [rule.value for rule in self.supplementary_rules],
            "board": [
                [tile_registry.tile_to_json(tile) if tile is not None else None for tile in row]
                for row in self.board
            ],
            "starting_position": self.starting_position.to_json(),
            "next_tile": tile_registry.tile_to_json(self.next_tile) if self.next_tile is not None else None,
            "players": self.players,
            "meeples": self.meeples,
            "abbots": self.abbots,
            "big_meeples": self.big_meeples,
            "placed_meeples": [[meeple.to_json() for meeple in player_meeples] for player_meeples in self.placed_meeples],
            "scores": self.scores,
            "current_player": self.current_player,
            "phase": self.phase.value,
            "last_tile_action": self.last_tile_action.to_json(tile_registry) if self.last_tile_action is not None else None,
            "last_river_rotation": self.last_river_rotation.value
        }

        return data
    
    @staticmethod
    def from_json(data: Dict[str, Any]):
        state = CarcassonneGameState(empty=True)
        state.deck = [tile_registry.tile_from_json(tile) for tile in data["deck"]]
        state.supplementary_rules = [SupplementaryRule(rule) for rule in data["supplementary_rules"]]
        state.board = [
            [tile_registry.tile_from_json(tile) if tile is not None else None for tile in row]
            for row in data["board"]
        ]
        state.starting_position = Coordinate.from_json(data["starting_position"])
        state.next_tile = tile_registry.tile_from_json(data["next_tile"]) if data["next_tile"] is not None else None
        state.players = data["players"]
        state.meeples = data["meeples"].copy()
        state.abbots = data["abbots"].copy()
        state.big_meeples = data["big_meeples"].copy()
        state.placed_meeples = [[MeeplePosition.from_json(meeple) for meeple in player_meeples] for player_meeples in data["placed_meeples"]]
        state.scores = data["scores"].copy()
        state.current_player = data["current_player"]
        state.phase = GamePhase(data["phase"])
        state.last_tile_action = TileAction.from_json(data["last_tile_action"], tile_registry) if data["last_tile_action"] is not None else None
        state.last_river_rotation = Rotation(data["last_river_rotation"])

        return state

    def simple_copy(self):
        new_state = CarcassonneGameState()
        new_state.deck = self.deck.copy()
        new_state.supplementary_rules = self.supplementary_rules.copy()
        # board is not copied
        new_state.board = self.board
        new_state.starting_position = self.starting_position
        new_state.next_tile = self.next_tile
        new_state.players = self.players
        new_state.meeples = self.meeples.copy()
        new_state.abbots = self.abbots.copy()
        new_state.big_meeples = self.big_meeples.copy()
        new_state.placed_meeples = self.placed_meeples.copy()
        for i in range(len(new_state.placed_meeples)):
            new_state.placed_meeples[i] = self.placed_meeples[i].copy()

        new_state.scores = self.scores.copy()
        new_state.current_player = self.current_player
        new_state.phase = self.phase
        new_state.last_tile_action = self.last_tile_action
        new_state.last_river_rotation = self.last_river_rotation
        return new_state

    def get_tile(self, row: int, column: int):
        if row < 0 or column < 0:
            return None
        elif row >= len(self.board) or column >= len(self.board[0]):
            return None
        else:
            return self.board[row][column]

    def empty_board(self):
        for row in self.board:
            for column in row:
                if column is not None:
                    return False
        return True

    def is_terminated(self) -> bool:
        return self.next_tile is None

    def initialize_deck(self, tile_sets: [TileSet]):
        deck: [Tile] = []

        # The river
        if TileSet.THE_RIVER in tile_sets:
            deck.append(the_river_tiles["river_start"])

            new_tiles = []
            for card_name, count in the_river_tile_counts.items():
                if card_name == "river_start":
                    continue
                if card_name == "river_end":
                    continue

                for i in range(count):
                    new_tiles.append(the_river_tiles[card_name])

            random.shuffle(new_tiles)
            for tile in new_tiles:
                deck.append(tile)

            deck.append(the_river_tiles["river_end"])

        new_tiles = []

        if TileSet.BASE in tile_sets:
            for card_name, count in base_tile_counts.items():
                for i in range(count):
                    new_tiles.append(base_tiles[card_name])

        if TileSet.INNS_AND_CATHEDRALS in tile_sets:
            for card_name, count in inns_and_cathedrals_tile_counts.items():
                for i in range(count):
                    new_tiles.append(inns_and_cathedrals_tiles[card_name])

        random.shuffle(new_tiles)
        for tile in new_tiles:
            deck.append(tile)

        if debug: print("[debug] carcassonne_game_state.py:100 6f98f3 len(deck) = {}".format(len(deck)))

        return deck
