from ..carcassonne_game_state import CarcassonneGameState, GamePhase
from ..objects.actions.action import Action
from ..objects.actions.pass_action import PassAction
from ..objects.actions.tile_action import TileAction
from ..objects.playing_position import PlayingPosition
from ..utils.possible_move_finder import PossibleMoveFinder
from ..utils.tile_position_finder import TilePositionFinder


class ActionUtil:

    @staticmethod
    def get_possible_actions(state: CarcassonneGameState):
        actions: [Action] = []
        if state.phase == GamePhase.TILES:
            possible_playing_positions: [PlayingPosition] = TilePositionFinder.possible_playing_positions(
                game_state=state,
                tile_to_play=state.next_tile
            )
            if len(possible_playing_positions) == 0:
                # print('No possible playing positions')
                # input()
                actions.append(PassAction())
            else:
                playing_position: PlayingPosition
                for playing_position in possible_playing_positions:
                    action = TileAction(
                        tile=state.next_tile.turn(playing_position.turns),
                        coordinate=playing_position.coordinate,
                        tile_rotations=playing_position.turns
                    )
                    actions.append(action)
        elif state.phase == GamePhase.MEEPLES:
            possible_meeple_actions = PossibleMoveFinder.possible_meeple_actions(game_state=state)
            actions.extend(possible_meeple_actions)
            actions.append(PassAction())
        return actions

