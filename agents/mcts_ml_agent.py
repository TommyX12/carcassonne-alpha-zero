from abc import abstractmethod
from dataclasses import dataclass, field
import os
import random

import numpy as np
from tqdm import tqdm

from games.game_state import GameState
from utils.context import Context

from .mcts import MCTS, MCTSConfig
from utils.utils import ModelSummarizer, load_model, load_optimizer, normalize_dict, save_model, save_optimizer
from .ml_agent import MLAgent
from typing import Any, Dict, List, Optional, Tuple, TypeVar
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from keras import layers, Model
from utils.debug import debug

@dataclass
class MCTSMLAgentConfig:
    train_epochs: int = 20
    train_batch_size: int = 128
    mcts_config: MCTSConfig = field(default_factory=MCTSConfig)
    learning_rate: float = 1e-3
    gradient_clipping: float = -1

TState = TypeVar('TState', bound=GameState)

class MCTSMLAgent(MLAgent[TState]):
    def __init__(self, name: str, model: Model, summary_writer, config: Optional[MCTSMLAgentConfig]=None):
        if config is None:
            config = MCTSMLAgentConfig()

        self.model = model
        self.name = name
        self.summary_writer = summary_writer
        self.config = config

        self.model_summarizer = ModelSummarizer(self.model)
        self._build_tf_function()

        self.metrics = Context()
        self.metrics.policy_loss = keras.metrics.Mean(name='policy_loss')
        self.metrics.value_loss = keras.metrics.Mean(name='value_loss')
        self.metrics.loss = keras.metrics.Mean(name='loss')

        self.reset_inference()

    def reset_inference(self):
        self.mcts = self._prepare_mcts()

    def _get_loss(self, pred_policy_logit, pred_value, target_policy, target_value):
        batch_size = tf.shape(target_value)[0]
        policy_loss = - \
            tf.reduce_sum(target_policy *
                            tf.nn.log_softmax(pred_policy_logit))
        # BE CAREFUL: predicted value has shape (batch_size, 1), but target value has shape (batch_size,)
        value_loss = tf.reduce_sum(
            tf.square(tf.reshape(target_value, (batch_size, 1)) -
                        tf.reshape(pred_value, (batch_size, 1))))
        loss = policy_loss + value_loss
        return policy_loss, value_loss, loss

    def _build_tf_function(self):
        optimizer_param = {
            'learning_rate': self.config.learning_rate,
        }
        if self.config.gradient_clipping > 0:
            optimizer_param['clipvalue'] = self.config.gradient_clipping

        self.optimizer = keras.optimizers.Adam(**optimizer_param)
        
        @tf.function
        def inference(state_input):
            return self.model(state_input, training=False)  # type: ignore

        self.inference = inference

        @tf.function
        def train_step(state_input, target_policy, target_value):
            with tf.GradientTape() as tape:
                batch_size = tf.shape(target_value)[0]
                model_output: Any = self.model(
                    state_input, training=True)  # type: ignore
                # shape = (batch_size, ACTION_TOTAL_DIM)
                pred_policy_logit = model_output['policy_logit']
                # shape = (batch_size, 1)
                pred_value = model_output['value']

                policy_loss, value_loss, loss = self._get_loss(pred_policy_logit, pred_value, target_policy, target_value)

                self.metrics.policy_loss(policy_loss)
                self.metrics.value_loss(value_loss)
                self.metrics.loss(loss)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            return loss

        self.train_step = train_step

    def get_action_probs(self, state: TState, temperature: float, add_exploration_noise: bool) -> Dict[Any, float]:
        return self.mcts.get_action_probs(state, temperature, add_exploration_noise)

    def _prepare_mcts(self):
        def inference_fn(state, all_legal_actions):
            if debug:
                print("[debug] mcts_ml_agent.py:329 d7a2f8 performing inference")
            if debug:
                print("[debug] mcts_ml_agent.py:330 27596f all_legal_actions = {}".format(
                    all_legal_actions))
            state_input = self._state_to_model_input(state)
            # expand batch dim
            state_input = {k: tf.expand_dims(v, 0)
                           for k, v in state_input.items()}
            if debug:
                for k, v in state_input.items():
                    print("[debug] mcts_ml_agent.py:335 3a4f4a state_input[{}] = {}".format(
                        k, tf.shape(v)))
            with self.model_summarizer:
                model_output: Any = self.inference(state_input)
            if debug:
                for k, v in model_output.items():
                    print("[debug] mcts_ml_agent.py:335 3a4f4a model_output[{}] = {}".format(
                        k, tf.shape(v)))
            policy = self._pred_policy_to_action_probs(
                model_output['policy_logit'][0], all_legal_actions)
            value = model_output['value'][0, 0].numpy()
            if debug:
                print(
                    "[debug] mcts_ml_agent.py:338 735776 policy = {}".format(policy))
            if debug:
                print(
                    "[debug] mcts_ml_agent.py:339 927d84 value = {}".format(value))
            return {'policy': policy, 'value': value}

        return MCTS(inference_fn, self.config.mcts_config)

    def save(self, path):
        save_model(self.model, path)
        # TODO: get optimizer saving working
        # save_optimizer(self.optimizer, self.model, path)

    def load(self, path):
        self.model = load_model(path)
        # load_optimizer(self.optimizer, self.model, path)
        self._build_tf_function()
        self.model_summarizer = ModelSummarizer(self.model)
        self.reset_inference()

    def train(self, train_data: List[Tuple[TState, Dict[Any, float], float]], current_iteration: int):
        """
        train_data may be shuffled in place.
        current_iteration is for logging purposes only.
        """

        def process_train_data(slice):
            state_input: Dict[str, Any] = {}
            target_policy = []
            target_value = []
            for state, policy, value in slice:
                model_input = self._state_to_model_input(state)
                for k, v in model_input.items():
                    if k not in state_input:
                        state_input[k] = []
                    state_input[k].append(v)

                target_policy.append(self._action_probs_to_label(policy))
                target_value.append(float(value))

            for k in state_input.keys():
                state_input[k] = tf.stack(state_input[k])

            target_policy = tf.stack(target_policy)
            target_value = tf.stack(target_value)

            return state_input, target_policy, target_value

        # data already shuffled
        # dataset = tf.data.Dataset.from_tensor_slices(
        #     (state_input, target_policy, target_value))
        # dataset = dataset.batch(self.config.train_batch_size)
        for epoch in tqdm(range(self.config.train_epochs), desc='Epoch', mininterval=1):
            for metric_name in self.metrics.keys():
                self.metrics[metric_name].reset_states()

            random.shuffle(train_data)
            for i in tqdm(range(0, len(train_data), self.config.train_batch_size), desc='Training', mininterval=1, leave=False):
                state_input, target_policy, target_value = process_train_data(
                    train_data[i:min(i + self.config.train_batch_size, len(train_data))])
                with self.model_summarizer:
                    loss = self.train_step(
                        state_input, target_policy, target_value)
                    tf.debugging.check_numerics(loss, 'Loss is inf or nan')

            for metric_name in self.metrics.keys():
                with self.summary_writer.as_default():
                    tf.summary.scalar(
                        f'{self.name}_{metric_name}', self.metrics[metric_name].result(), step=current_iteration * self.config.train_epochs + epoch)

    @abstractmethod
    def _state_to_model_input(self, state: TState) -> Dict[str, Tensor]:
        """
        Return a dict of tensors. The tensors should not have batch dim.
        """
        pass

    @abstractmethod
    def _pred_policy_to_action_probs(self, pred_policy_logit, game_legal_actions) -> Dict[Any, float]:
        """
        Guaranteed to return all legal actions. MCTS will use this.
        pred_policy should be a tensor of shape (ACTION_TOTAL_DIM,)
        """
        pass

    @abstractmethod
    def _action_probs_to_label(self, action_probs: Dict[Any, float]):
        """
        Convert action probs to label for training.
        Return a tensor of shape = (action_dim,).
        """
        pass

