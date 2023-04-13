

from .game import Game
from .othello_game_state import OthelloGameState


class OthelloGame(Game[OthelloGameState]):
    def get_initial_state(self) -> OthelloGameState:
        n = 6
        state = OthelloGameState(n)
        state.board[int(n/2)-1][int(n/2)] = 1
        state.board[int(n/2)][int(n/2)-1] = 1
        state.board[int(n/2)-1][int(n/2)-1] = -1
        state.board[int(n/2)][int(n/2)] = -1
        return state
