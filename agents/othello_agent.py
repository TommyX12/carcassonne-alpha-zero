from collections import OrderedDict
import os

import numpy as np

from .fake_td_mcts import FakeTDMCTS
from .mcts import MCTSConfig
from .third_party_mcts import ThirdPartyMCTS

from games.othello_game_state import OthelloGameState, PassAction

from .mcts_ml_agent import MCTSMLAgent, MCTSMLAgentConfig
from utils.utils import ModelSummarizer, load_model, load_optimizer, normalize_dict, process_offset_dict, save_model, save_optimizer
from typing import Any, Dict, List, Tuple
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from keras import layers, Model
from utils.debug import debug


BOARD_SIZE = 6

TILE_FEATURE_TOTAL_DIM = 1

ACTION_OFFSETS: OrderedDict[str, int] = OrderedDict()
TILE_ACTION_DIM = BOARD_SIZE * BOARD_SIZE
ACTION_OFFSETS['place_piece'] = TILE_ACTION_DIM
ACTION_OFFSETS['pass'] = 1

ACTION_TOTAL_DIM = process_offset_dict(ACTION_OFFSETS)
OTHER_ACTION_DIM = ACTION_TOTAL_DIM - TILE_ACTION_DIM


def action_to_idx(action: Any):
    if isinstance(action, PassAction):
        return ACTION_OFFSETS['pass']

    return action[0] * BOARD_SIZE + action[1]


LATENT_DIM = 128
BACKBONE_LATENT_DIM = 128
DROPOUT = 0.2
POLICY_TILE_HEAD_LATENT_DIM = 128
VALUE_HEAD_INIT_STDDEV = 0.01


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
            # DownConvLayer(),
            # DownConvLayer(),
        ]

    def call(self, x):
        intermediate_features = []
        for layer in self.sequence:
            intermediate_features.append(x)
            x = layer(x)

        return x, intermediate_features


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
            # UpConvLayer(),
            # UpConvLayer(),
            layers.Conv2DTranspose(1, 3),
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
        return self.sequence(x)


class OthelloModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = tf.keras.Sequential([
            layers.Conv2D(512, 3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(512, 3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(512, 3, strides=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(512, 3, strides=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.3),
            layers.Reshape((-1,)),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.3),
        ])

        self.fc3 = layers.Dense(ACTION_TOTAL_DIM)
        self.fc4 = layers.Dense(1)

    def call(self, x):
        # shape = (batch_size, BOARD_SIZE, BOARD_SIZE, TILE_FEATURE_TOTAL_DIM)
        board = x['board']

        feature = self.backbone(board)

        policy_logit = self.fc3(feature)
        value = tf.math.tanh(self.fc4(feature))

        return {'policy_logit': policy_logit, 'value': value}


class OthelloAgent(MCTSMLAgent[OthelloGameState]):
    def __init__(self, name: str, summary_writer):
        self.use_third_party_mcts = False
        super().__init__(name, OthelloModel(), summary_writer, config=MCTSMLAgentConfig(
            train_epochs = 10,
            train_batch_size = 64,
            gradient_clipping=-1,
            mcts_config = MCTSConfig(
                num_search=25,
                enable_exploration_noise=False,
                enable_value_normalization=False,
            ),
        ))

    def _get_loss(self, pred_policy_logit, pred_value, target_policy, target_value):
        batch_size = tf.shape(target_value)[0]
        policy_loss = - \
            tf.reduce_mean(tf.reduce_sum(target_policy *
                            tf.nn.log_softmax(pred_policy_logit), axis=1), axis=0)
        # BE CAREFUL: predicted value has shape (batch_size, 1), but target value has shape (batch_size,)
        value_loss = tf.reduce_mean(
            tf.square(tf.reshape(target_value, (batch_size, 1)) -
                        tf.reshape(pred_value, (batch_size, 1))))
        loss = policy_loss + value_loss
        return policy_loss, value_loss, loss

    def _prepare_mcts(self):
        # def inference_fn(state, all_legal_actions):
        #     state_input = self._state_to_model_input(state)
        #     # expand batch dim
        #     state_input = {k: tf.expand_dims(v, 0)
        #                    for k, v in state_input.items()}
        #     with self.model_summarizer:
        #         model_output: Any = self.inference(state_input)
        #     policy = self._pred_policy_to_action_probs(
        #         model_output['policy_logit'][0], all_legal_actions)
        #     value = model_output['value'][0, 0].numpy()
        #     return {'policy': policy, 'value': value}

        # return FakeTDMCTS(inference_fn)

        if self.use_third_party_mcts:
            def inference_fn(state):
                state_input = self._state_to_model_input(state)
                # expand batch dim
                state_input = {k: tf.expand_dims(v, 0)
                            for k, v in state_input.items()}
                with self.model_summarizer:
                    model_output: Any = self.inference(state_input)
                return tf.nn.softmax(model_output['policy_logit'][0]).numpy(), model_output['value'][0, 0].numpy()

            return ThirdPartyMCTS(inference_fn, self.config.mcts_config)

        return super()._prepare_mcts()

    def _state_to_model_input(self, state: OthelloGameState) -> Dict[str, Tensor]:
        board_input = np.zeros((BOARD_SIZE, BOARD_SIZE, TILE_FEATURE_TOTAL_DIM))

        color = state._player_to_color(state.current_player)

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state.board[i][j] == color:
                    board_input[i][j][0] = 1
                
                elif state.board[i][j] == -color:
                    board_input[i][j][0] = -1

        result: Dict[str, Tensor] = {}
        result['board'] = tf.convert_to_tensor(board_input, dtype=tf.float32)

        return result

    def _pred_policy_to_action_probs(self, pred_policy_logit, game_legal_actions) -> Dict[Any, float]:
        """
        Guaranteed to return all legal actions. MCTS will use this.
        pred_policy should be a tensor of shape (ACTION_TOTAL_DIM,)
        """
        assert len(game_legal_actions) > 0
        pred_policy = tf.nn.softmax(pred_policy_logit)

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
