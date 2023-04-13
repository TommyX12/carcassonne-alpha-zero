from typing import Any

from carcassonne.utils.state_updater import StateUpdater
from carcassonne.utils.action_util import ActionUtil
from .game_state import GameState
from carcassonne.carcassonne_game_state import CarcassonneGameState as LibCarcassonneGameState
from carcassonne.carcassonne_visualiser import CarcassonneVisualiser


visualizer = CarcassonneVisualiser()


class CarcassonneGameState(GameState):
    def __init__(self, lib_state: LibCarcassonneGameState):
        assert lib_state.players == 2
        self.lib_state: LibCarcassonneGameState = lib_state

    def get_legal_actions(self):
        return ActionUtil.get_possible_actions(self.lib_state)

    def is_terminal(self) -> bool:
        return self.lib_state.is_terminated()

    def get_num_players(self) -> int:
        return self.lib_state.players

    def get_current_player(self) -> int:
        return self.lib_state.current_player

    def get_player_value(self, player: int) -> float:
        diff = self.lib_state.scores[player] - self.lib_state.scores[1 - player]
        if diff > 0:
            return 1
        
        if diff < 0:
            return -1

        return 0

    def get_player_score(self, player: int) -> float:
        return self.lib_state.scores[player]

    def apply_action(self, action: Any) -> 'GameState':
        return CarcassonneGameState(StateUpdater.apply_action(self.lib_state, action))

    def visualize(self):
        visualizer.draw_game_state(self.lib_state)

    def to_json(self):
        return self.lib_state.to_json()
    
    @staticmethod
    def from_json(data):
        return CarcassonneGameState(LibCarcassonneGameState.from_json(data))

    def get_reuse_hash_key(self) -> Any:
        return None