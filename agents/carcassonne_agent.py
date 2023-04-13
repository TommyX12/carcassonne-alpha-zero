from collections import OrderedDict
import os

import numpy as np

from .mcts_ml_agent import MCTSMLAgent
from carcassonne.objects.actions.tile_action import TileAction
from carcassonne.objects.actions.meeple_action import MeepleAction
from carcassonne.objects.actions.pass_action import PassAction
from carcassonne.objects.actions.action import Action
from utils.utils import ModelSummarizer, load_model, load_optimizer, normalize_dict, process_offset_dict, save_model, save_optimizer
from carcassonne.objects.game_phase import GamePhase
from carcassonne.objects.tile import Tile
from games.carcassonne_game_state import CarcassonneGameState
from typing import Any, Dict, List, Tuple
import tensorflow as tf
from tensorflow import Tensor
from carcassonne.tile_sets.base_deck import base_tiles
from carcassonne.tile_sets.inns_and_cathedrals_deck import inns_and_cathedrals_tiles
from tensorflow import keras
from keras import layers, Model
from utils.debug import debug


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(FILE_PATH, '../carcassonne/resources/images')

TILE_FEATURE_DIM = 576  # 1152
NUM_MEEPLE_TYPES = 5
NUM_MEEPLE_SIDES = 9
TOTAL_MEEPLES = 7
TOTAL_ABBOTS = 1
TOTAL_BIG_MEEPLES = 1


TILE_FEATURE_OFFSETS: OrderedDict[str, int] = OrderedDict()
# Start by writing the feature dimensions. We will replace them with offsets later.
TILE_FEATURE_OFFSETS['resnet'] = TILE_FEATURE_DIM
TILE_FEATURE_OFFSETS['just_placed'] = 1
TILE_FEATURE_OFFSETS['is_meeple_friendly'] = 1
TILE_FEATURE_OFFSETS['meeple_type'] = NUM_MEEPLE_TYPES
TILE_FEATURE_OFFSETS['meeple_side'] = NUM_MEEPLE_SIDES

TILE_FEATURE_TOTAL_DIM = process_offset_dict(TILE_FEATURE_OFFSETS)

GAME_FEATURE_OFFSETS: OrderedDict[str, int] = OrderedDict()
GAME_FEATURE_OFFSETS['next_tile_resnet_0'] = TILE_FEATURE_DIM
GAME_FEATURE_OFFSETS['next_tile_resnet_1'] = TILE_FEATURE_DIM
GAME_FEATURE_OFFSETS['next_tile_resnet_2'] = TILE_FEATURE_DIM
GAME_FEATURE_OFFSETS['next_tile_resnet_3'] = TILE_FEATURE_DIM
GAME_FEATURE_OFFSETS['player_meeples'] = TOTAL_MEEPLES
GAME_FEATURE_OFFSETS['player_abbots'] = TOTAL_ABBOTS
GAME_FEATURE_OFFSETS['player_big_meeples'] = TOTAL_BIG_MEEPLES
GAME_FEATURE_OFFSETS['enemy_meeples'] = TOTAL_MEEPLES
GAME_FEATURE_OFFSETS['enemy_abbots'] = TOTAL_ABBOTS
GAME_FEATURE_OFFSETS['enemy_big_meeples'] = TOTAL_BIG_MEEPLES
GAME_FEATURE_OFFSETS['is_meeples_phase'] = 1
GAME_FEATURE_OFFSETS['current_value'] = 1

GAME_FEATURE_TOTAL_DIM = process_offset_dict(GAME_FEATURE_OFFSETS)

BOARD_SIZE = 10

ACTION_OFFSETS: OrderedDict[str, int] = OrderedDict()
TILE_ACTION_DIM = BOARD_SIZE * BOARD_SIZE * 4
ACTION_OFFSETS['place_tile'] = TILE_ACTION_DIM
ACTION_OFFSETS['place_meeple'] = NUM_MEEPLE_SIDES * NUM_MEEPLE_TYPES
ACTION_OFFSETS['remove_abbot'] = 1
ACTION_OFFSETS['pass'] = 1

ACTION_TOTAL_DIM = process_offset_dict(ACTION_OFFSETS)
OTHER_ACTION_DIM = ACTION_TOTAL_DIM - TILE_ACTION_DIM


def action_to_idx(action: Action):
    if isinstance(action, PassAction):
        return ACTION_OFFSETS['pass']

    if isinstance(action, TileAction):
        return ACTION_OFFSETS['place_tile'] + action.coordinate.row * BOARD_SIZE * 4 + action.coordinate.column * 4 + action.tile_rotations

    if isinstance(action, MeepleAction):
        if action.remove:
            return ACTION_OFFSETS['remove_abbot']

        return ACTION_OFFSETS['place_meeple'] + action.coordinate_with_side.side * NUM_MEEPLE_TYPES + action.meeple_type

    raise Exception('Unknown action type: ' + str(action))


class TileFeatureProvider(object):
    def __init__(self):
        super().__init__()
        self.cache = {}
        # prepopulate using known tiles
        for tile in base_tiles.values():
            self.get_tile_feature(tile)

        for tile in inns_and_cathedrals_tiles.values():
            self.get_tile_feature(tile)

    def get_tile_feature(self, tile: Tile) -> Tensor:
        id = tile.description
        if id not in self.cache:
            features = np.load(os.path.join(
                IMAGE_DIR, tile.image + '.feat.npy'))
            assert len(features) == 4
            assert len(features[0]) == TILE_FEATURE_DIM
            self.cache[id] = features

        return self.cache[id][tile.turns]


tile_feature_provider = TileFeatureProvider()

LATENT_DIM = 128
BACKBONE_LATENT_DIM = 128
DROPOUT = 0.2
GAME_FEATURE_LATENT_DIM = 256
POLICY_TILE_HEAD_LATENT_DIM = 128
VALUE_HEAD_INIT_STDDEV = 0.01

BOOL_FEATURE_SCALE = 20.0


class DownConvLayer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence = tf.keras.Sequential([
            layers.Conv2D(LATENT_DIM, 3),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
        ])

    def call(self, x):
        return self.sequence(x)


class UpConvLayer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence = tf.keras.Sequential([
            layers.Conv2DTranspose(POLICY_TILE_HEAD_LATENT_DIM, 3),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
        ])

    def call(self, x):
        return self.sequence(x)


class BoardEncoder(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence = [
            DownConvLayer(),
            DownConvLayer(),
            DownConvLayer(),
            DownConvLayer(),
        ]

    def call(self, x):
        intermediate_features = []
        for layer in self.sequence:
            intermediate_features.append(x)
            x = layer(x)

        return x, intermediate_features


class GameEncoder(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence = tf.keras.Sequential([
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(GAME_FEATURE_LATENT_DIM),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
        ])

    def call(self, x):
        return self.sequence(x)


class Backbone(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence = tf.keras.Sequential([
            layers.Conv2D(BACKBONE_LATENT_DIM, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dropout(DROPOUT),
            layers.Conv2D(BACKBONE_LATENT_DIM, 1),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dropout(DROPOUT),
        ])

    def call(self, x):
        return self.sequence(x)


class PolicyHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tile_action_head = [
            UpConvLayer(),
            UpConvLayer(),
            UpConvLayer(),
            layers.Conv2DTranspose(4, 3),
        ]
        self.other_action_head = tf.keras.Sequential([
            layers.Reshape((-1,)),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(OTHER_ACTION_DIM),
        ])

    def call(self, x, intermediate_features):
        batch_size = tf.shape(x)[0]
        other_action = self.other_action_head(x)
        for i in range(len(self.tile_action_head)):
            layer = self.tile_action_head[i]
            if i > 0:
                x = tf.concat([x, intermediate_features[-i]], axis=-1)

            x = layer(x)

        tile_action = x
        return tf.concat([tf.reshape(tile_action, (batch_size, -1)), other_action], axis=-1)


class ValueHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence = tf.keras.Sequential([
            layers.Reshape((-1,)),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(1),
        ])

    def call(self, x):
        return tf.math.tanh(self.sequence(x))


class CarcassonneModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.board_encoder = BoardEncoder()
        self.game_encoder = GameEncoder()
        self.backbone = Backbone()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def call(self, x):
        # shape = (batch_size, 10, 10, TILE_FEATURE_TOTAL_DIM)
        board = x['board']
        # shape = (batch_size, GAME_FEATURE_TOTAL_DIM)
        game = x['game']
        batch_size = tf.shape(game)[0]

        # current value
        # offset = GAME_FEATURE_OFFSETS['current_value']
        # current_value = game[:, offset:offset+1]

        # shape = (batch_size, 2, 2, LATENT_DIM)
        board_feature, intermediate_features = self.board_encoder(
            board)  # type: ignore

        # shape = (batch_size, GAME_FEATURE_LATENT_DIM)
        game_feature = self.game_encoder(game)

        game_feature = tf.reshape(
            game_feature, (batch_size, 1, 1, GAME_FEATURE_LATENT_DIM))
        game_feature = tf.tile(game_feature, (1, 2, 2, 1))
        feature = tf.concat([board_feature, game_feature], axis=-1)

        feature = self.backbone(feature)

        policy_logit = self.policy_head(feature, intermediate_features)
        value = self.value_head(feature)

        return {'policy_logit': policy_logit, 'value': value}


class CarcassonneAgent(MCTSMLAgent[CarcassonneGameState]):
    def __init__(self, name: str, summary_writer):
        super().__init__(name, CarcassonneModel(), summary_writer)

    def _state_to_model_input(self, state: CarcassonneGameState) -> Dict[str, Tensor]:
        lib_state = state.lib_state
        player = state.get_current_player()
        result: Dict[str, Tensor] = {}
        # remember to canonicalize the state with respect to the current player
        # current board state. tile features:
        num_rows = len(lib_state.board)
        num_columns = len(lib_state.board[0])
        assert num_rows == BOARD_SIZE and num_columns == BOARD_SIZE
        board_input = np.zeros((num_rows, num_columns, TILE_FEATURE_TOTAL_DIM))
        for row in range(num_rows):
            for column in range(num_columns):
                tile = lib_state.board[row][column]
                if tile is None:
                    continue

                offset = TILE_FEATURE_OFFSETS['resnet']
                feature = tile_feature_provider.get_tile_feature(tile)
                if debug.tile_feature:
                    print("[debug] carcassonne_agent.py:403 dd4233 tile({},{}) = {}".format(
                        row, column, feature[:10]))
                board_input[row][column][offset:offset +
                                         TILE_FEATURE_DIM] = feature

        last_tile_action = lib_state.last_tile_action
        if last_tile_action is not None:
            row = last_tile_action.coordinate.row
            column = last_tile_action.coordinate.column
            offset = TILE_FEATURE_OFFSETS['just_placed']
            board_input[row][column][offset] = BOOL_FEATURE_SCALE

        for meeple_player, meeples in enumerate(lib_state.placed_meeples):
            is_friendly = int(meeple_player == player) * BOOL_FEATURE_SCALE
            for meeple in meeples:
                row = meeple.coordinate_with_side.coordinate.row
                column = meeple.coordinate_with_side.coordinate.column
                offset = TILE_FEATURE_OFFSETS['is_meeple_friendly']
                board_input[row][column][offset] = is_friendly
                offset = TILE_FEATURE_OFFSETS['meeple_type']
                board_input[row][column][offset +
                                         meeple.meeple_type] = BOOL_FEATURE_SCALE
                offset = TILE_FEATURE_OFFSETS['meeple_side']
                board_input[row][column][offset +
                                         meeple.coordinate_with_side.side] = BOOL_FEATURE_SCALE

        game_input = np.zeros((GAME_FEATURE_TOTAL_DIM))

        # tile about to be placed (i.e. next tile)
        if lib_state.next_tile is not None:
            if debug.tile_feature:
                print("[debug] carcassonne_agent.py:403 dd4233 next_tile = {}".format(
                    lib_state.next_tile.description))

            for rotation in range(4):
                offset = GAME_FEATURE_OFFSETS[f'next_tile_resnet_{rotation}']
                feature = tile_feature_provider.get_tile_feature(
                    lib_state.next_tile.turn(rotation))
                game_input[offset:offset+TILE_FEATURE_DIM] = feature
                if debug.tile_feature:
                    print("[debug] carcassonne_agent.py:403 dd4233 next_tile_feature({}) = {}".format(
                        rotation, feature[:10]))

        # player meeple inventory
        offset = GAME_FEATURE_OFFSETS['player_meeples']
        for i in range(lib_state.meeples[player]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['player_abbots']
        for i in range(lib_state.abbots[player]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['player_big_meeples']
        for i in range(lib_state.big_meeples[player]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        # enemy meeple inventory
        enemy = 1 - player
        offset = GAME_FEATURE_OFFSETS['enemy_meeples']
        for i in range(lib_state.meeples[enemy]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['enemy_abbots']
        for i in range(lib_state.abbots[enemy]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['enemy_big_meeples']
        for i in range(lib_state.big_meeples[enemy]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        # is meeples phase
        offset = GAME_FEATURE_OFFSETS['is_meeples_phase']
        game_input[offset] = int(lib_state.phase == GamePhase.MEEPLES) * BOOL_FEATURE_SCALE

        # current player value
        offset = GAME_FEATURE_OFFSETS['current_value']
        game_input[offset] = float(state.get_player_value(player))

        result['board'] = tf.convert_to_tensor(board_input, dtype=tf.float32)
        result['game'] = tf.convert_to_tensor(game_input, dtype=tf.float32)

        return result

    def _pred_policy_to_action_probs(self, pred_policy_logit, game_legal_actions) -> Dict[Any, float]:
        """
        Guaranteed to return all legal actions. MCTS will use this.
        pred_policy should be a tensor of shape (ACTION_TOTAL_DIM,)
        """
        if debug:
            print(
                "[debug] carcassonne_agent.py:463 4de025 _pred_policy_to_action_probs")
        assert len(game_legal_actions) > 0
        pred_policy = tf.nn.softmax(pred_policy_logit)
        if debug:
            for action in game_legal_actions:
                if action_to_idx(action) >= 442:
                    print(
                        "[debug] carcassonne_agent.py:471 51c9f2 action = {}".format(action))

        game_legal_action_indices = [action_to_idx(
            action) for action in game_legal_actions]
        pred_policy_numpy = pred_policy.numpy()  # type: ignore
        game_legal_action_probs = pred_policy_numpy[game_legal_action_indices]

        policy = {
            action: prob
            for action, prob in zip(game_legal_actions, game_legal_action_probs)
        }
        # re-normalize
        policy = normalize_dict(policy)
        if policy is None:
            # model did not return any possible legal actions
            # fallback to uniform policy
            policy = {
                action: 1.0 / len(game_legal_actions)
                for action in game_legal_actions
            }

        if debug:
            print("[debug] carcassonne_agent.py:465 878824 policy = {}".format(policy))

        return policy

    def _action_probs_to_label(self, action_probs: Dict[Any, float]):
        """
        Convert action probs to label for training.
        Return a tensor.
        """
        label = np.zeros((ACTION_TOTAL_DIM,))
        for action, prob in action_probs.items():
            label[action_to_idx(action)] = prob

        return tf.convert_to_tensor(label, dtype=tf.float32)
