
from .game_state import GameState
from typing import Any, List, Optional, Tuple
import numpy as np

square_content = {
    -1: "X",
    +0: "-",
    +1: "O"
}

all_directions = [(1, 1), (1, 0), (1, -1), (0, -1),
                  (-1, -1), (-1, 0), (-1, 1), (0, 1)]


class PassAction(object):
    def __eq__(self, other):
        return isinstance(other, PassAction)

    def __hash__(self):
        return 42069


class OthelloGameState(GameState):
    def __init__(self, n, board: Optional[np.ndarray] = None, current_player=0):
        self.n = n
        if board is None:
            board = np.zeros((n, n), dtype=np.float32)

        self.board = board
        self.current_player = current_player

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(lambda x: x[0] + x[1], zip(move, direction)))
        # move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)):
            # while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move = list(map(lambda x: x[0] + x[1], zip(move, direction)))
            # move = (move[0]+direction[0],move[1]+direction[1])

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self.board[x][y]
        flips = []

        for x, y in self._increment_move(origin, direction, self.n):
            if self.board[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self.board[x][y] == color:
                return None
            elif self.board[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_moves_for_square(self, square: Tuple[int, int]):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        square: (x,y)
        """
        (x, y) = square

        # determine the color of the piece.
        color = self.board[x][y]

        # skip empty source squares.
        if color == 0:
            return []

        # search all possible directions.
        moves = []
        for direction in all_directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def _player_to_color(self, player):
        return 1 if player == 0 else -1

    def _color_to_player(self, color):
        return 0 if color == 1 else 1

    def get_legal_actions(self) -> Any:
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        """
        moves = set()  # stores the legal moves.

        color = self._player_to_color(self.current_player)

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self.board[x][y] == color:
                    newmoves = self._get_moves_for_square((x, y))
                    moves.update(newmoves)

        if len(moves) == 0:
            return [PassAction()]

        return moves

    def _has_legal_actions(self, player) -> bool:
        color = self._player_to_color(player)

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self.board[x][y] == color:
                    newmoves = self._get_moves_for_square((x, y))
                    if len(newmoves) > 0:
                        return True

        return False

    def is_terminal(self) -> bool:
        return not (self._has_legal_actions(0) or self._has_legal_actions(1))

    def get_num_players(self) -> int:
        return 2

    def get_current_player(self) -> int:
        return self.current_player

    def _count_diff(self, player):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        color = self._player_to_color(player)
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self.board[x][y] == color:
                    count += 1
                if self.board[x][y] == -color:
                    count -= 1

        return count

    def get_player_value(self, player: int) -> float:
        diff = self._count_diff(player)
        if diff > 0:
            return 1
        else:
            return -1

    def get_player_score(self, player: int) -> float:
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        color = self._player_to_color(player)
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self.board[x][y] == color:
                    count += 1

        return count

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        # initialize variables
        flips = [origin]

        for x, y in self._increment_move(origin, direction, self.n):
            # print(x,y)
            if self.board[x][y] == 0:
                return []
            if self.board[x][y] == -color:
                flips.append((x, y))
            elif self.board[x][y] == color and len(flips) > 0:
                # print(flips)
                return flips

        return []

    def apply_action(self, action: Any) -> 'GameState':
        next_player = 1 - self.current_player
        if isinstance(action, PassAction):
            return OthelloGameState(self.n, self.board, next_player)

        board = np.copy(self.board)
        color = self._player_to_color(self.current_player)
        flips = [flip for direction in all_directions
                 for flip in self._get_flips(action, direction, color)]
        assert len(list(flips)) > 0
        for x, y in flips:
            board[x][y] = color

        return OthelloGameState(self.n, board, next_player)

    def visualize(self):
        n = self.board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = self.board[y][x]    # get the piece to print
                print(square_content[piece], end=" ")
            print("|")

        print("-----------------------")

    # ---- compatibility methods for the AlphaZero implementation ----

    def get_string_representation(self):
        return (self.board * self._player_to_color(self.get_current_player())).tobytes()

    def get_action_size(self):
        return self.n * self.n + 1

    def get_valid_move_mask(self):
        mask = np.zeros(self.get_action_size())
        for action in self.get_legal_actions():
            if isinstance(action, PassAction):
                mask[-1] = 1
            else:
                mask[action[0] * self.n + action[1]] = 1

        return mask

    def get_next_state(self, action):
        if action == self.get_action_size() - 1:
            return self.apply_action(PassAction())

        else:
            return self.apply_action((action // self.n, action % self.n))

    def get_reuse_hash_key(self):
        return (self.board.tobytes(), self.current_player)
