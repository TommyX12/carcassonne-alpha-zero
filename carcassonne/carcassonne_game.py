from .carcassonne_game_state import CarcassonneGameState
from .carcassonne_visualiser import CarcassonneVisualiser
from .objects.actions.action import Action
from .tile_sets.supplementary_rules import SupplementaryRule
from .tile_sets.tile_sets import TileSet
from .utils.action_util import ActionUtil
from .utils.state_updater import StateUpdater


class CarcassonneGame:

    def __init__(self,
                 players: int = 2,
                 tile_sets: [TileSet] = (TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS),
                 supplementary_rules: [SupplementaryRule] = (SupplementaryRule.FARMERS, SupplementaryRule.ABBOTS)):
        self.players = players
        self.tile_sets = tile_sets
        if TileSet.THE_RIVER in tile_sets:
            raise NotImplementedError("The river has bug related to get_river_rotation_ends (need to find the right starting end for previous_river_ends). Do not use")

        self.supplementary_rules = supplementary_rules
        self.state: CarcassonneGameState = CarcassonneGameState(
            tile_sets=tile_sets,
            players=players,
            supplementary_rules=supplementary_rules
        )
        self.visualiser = CarcassonneVisualiser()

    def reset(self):
        self.state = CarcassonneGameState(tile_sets=self.tile_sets, supplementary_rules=self.supplementary_rules)

    def step(self, player: int, action: Action):
        self.state = StateUpdater.apply_action(game_state=self.state, action=action)

    def render(self):
        self.visualiser.draw_game_state(self.state)

    def is_finished(self) -> bool:
        return self.state.is_terminated()

    def get_current_player(self) -> int:
        return self.state.current_player

    def get_possible_actions(self) -> [Action]:
        return ActionUtil.get_possible_actions(self.state)
