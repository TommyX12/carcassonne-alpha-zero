from collections import deque
import itertools
import json
import os
import random
import sys
from typing import Any, Dict, List
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path

from datetime import datetime

from tensorflow import keras

from sacred import Experiment
from sacred.observers import FileStorageObserver
from agents.agent import Agent
from agents.carcassonne_agent import CarcassonneAgent
from agents.ml_agent import MLAgent
from agents.random_agent import RandomAgent
from carcassonne.carcassonne_game_state import action_from_json, action_to_json
from carcassonne.utils.farm_util import FarmUtil
from games.carcassonne_game import CarcassonneGame
from games.carcassonne_game_state import CarcassonneGameState
from games.game import Game
from games.game_state import GameState

from tasks.common_tasks import JuliaSetTask, julia_iteration
from utils.context import Context, default, factory, patch
from utils.experiment import ex, init_experiment
from utils.utils import CheckpointDirManager, CyclicCounter, ModelGraphVisualizer, CheckpointManager, ModelSummarizer, argmax_dict, pretty_print_dict
from utils.debug import debug

from tqdm import tqdm


NAME = 'carcassonne'
init_experiment(NAME, Experiment(NAME))


@ex.sacred.config
def config():
    load_checkpoint_path = None
    num_iters = 100
    num_self_play_episodes = 100
    max_train_data_per_iter = num_self_play_episodes * 180
    max_train_data_iters = 4
    temperature_threshold = 60
    arena_iterations = 20
    arena_win_threshold = 0
    estimated_turns_per_game = 180 # for progress bar only
    random_arena_iterations = 20


@ex.sacred.automain
def run(experiment_dir, tensorboard_dir, load_checkpoint_path, num_iters, max_train_data_per_iter, num_self_play_episodes, temperature_threshold, max_train_data_iters, arena_iterations, arena_win_threshold, estimated_turns_per_game, random_arena_iterations):
    # Create the agent
    prev_agent: MLAgent = CarcassonneAgent('prev_agent', summary_writer=ex.summary_writer)
    prev_agent.model_summarizer.finished = True # do not summarize the previous model
    agent: MLAgent = CarcassonneAgent('new_agent', summary_writer=ex.summary_writer)
    random_agent: Agent = RandomAgent()
    # Create the game
    game: Game = CarcassonneGame()

    train_data_by_iter = []

    temp_save_path = os.path.join(experiment_dir, f'temp_checkpoint')
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoints_manager = CheckpointDirManager(checkpoints_dir)

    current_iteration_start = 0

    def train_data_by_iter_from_json(json_data: List[List[Dict[str, Any]]]):
        train_data_by_iter = []
        for iter in tqdm(json_data, desc="Loading train data iter", leave=False, mininterval=1):
            train_data_iter = []
            for entry_json_data in tqdm(iter, desc="Loading train data", leave=False, mininterval=1):
                state = CarcassonneGameState.from_json(entry_json_data['state'])
                action_probs = {}
                for action_json_data, prob in entry_json_data['action_probs']:
                    action = action_from_json(action_json_data)
                    action_probs[action] = prob

                value = entry_json_data['value']
                train_data_iter.append((state, action_probs, value))

            train_data_by_iter.append(train_data_iter)

        return train_data_by_iter

    def train_data_by_iter_to_json(train_data_by_iter):
        json_data = []
        for iter in train_data_by_iter:
            train_data_iter = []
            for state, action_probs, value in iter:
                entry_json_data = {
                    'state': state.to_json(),
                    'action_probs': [(action_to_json(action), prob) for action, prob in action_probs.items()],
                    'value': value
                }
                train_data_iter.append(entry_json_data)

            json_data.append(train_data_iter)

        return json_data

    def execute_episode(metrics_step):
        # input()
        i = 0
        state = game.get_initial_state()
        train_data = []
        agent.reset_inference()
        for _ in tqdm(itertools.count(), desc="Self play (episode)", total=estimated_turns_per_game, leave=False, mininterval=1, disable=False):
            i += 1
            # Execute the episode
            # Return the data
            temperature = int(i < temperature_threshold)
            action_probs: Dict[Any, float] = agent.get_action_probs(state, temperature=temperature, add_exploration_noise=True)
            train_data.append((state, action_probs))

            actions = list(action_probs.keys())
            probs = [action_probs[a] for a in actions]
            action = np.random.choice(actions, p=probs)
            # print('last action applied', action)
            # old_state = state
            # state.visualize()
            state = state.apply_action(action)
            # if state.is_terminal():
            #     print('ok')
            #     FarmUtil.print_things = True
            #     old_state.apply_action(action)
            # print('current player', state.get_current_player())
            # print('score for player 1', state.get_player_score(0))
            # print('score for player 2', state.get_player_score(1))

            if state.is_terminal():
                # old_state.visualize()
                # a = input()
                # if a == 'q':
                #     exit(1)
                value = state.get_player_value(state.get_current_player())
                data = [(s, p, value * ((-1) ** (state.get_current_player() != s.get_current_player()))) for s, p in train_data]

                player_1_score = state.get_player_score(0)
                player_2_score = state.get_player_score(1)
                with ex.summary_writer.as_default():
                    tf.summary.scalar('self_play_episode_player_1_score', player_1_score, step=metrics_step)
                    tf.summary.scalar('self_play_episode_player_2_score', player_2_score, step=metrics_step)

                # abnormal_score_detected = False
                # if player_1_score > 150:
                #     print('Abnormal score detected for player 1:', player_1_score)
                #     abnormal_score_detected = True

                # if player_2_score > 150:
                #     print('Abnormal score detected for player 2:', player_2_score)
                #     abnormal_score_detected = True

                # if abnormal_score_detected:
                #     for s, p in train_data:
                #         print('current player', s.get_current_player())
                #         print('score 0', s.get_player_score(0))
                #         print('score 1', s.get_player_score(1))
                #         s.visualize()
                #         input()

                if debug.self_play:
                    train_data[-2][0].visualize()
                    input()

                return data

        raise Exception("Not possible")

    def play_arena_games(game: Game, iterations, agent1: Agent, agent2: Agent):
        iterations = int(iterations / 2)
        win1 = 0
        win2 = 0
        draws = 0

        def play_game(players: List[Agent]):
            state: GameState = game.get_initial_state()
            for player in players:
                player.reset_inference()

            for _ in tqdm(itertools.count(), desc="Arena game", total=estimated_turns_per_game, leave=False, mininterval=1):
                if debug.arena:
                    print("[debug] carcassonne_train.py:93 c9d504 state.get_current_player() = {}".format(state.get_current_player()))
                    print('player 1 score:', state.lib_state.scores[0]) # type: ignore
                    print('player 2 score:', state.lib_state.scores[1]) # type: ignore
                    print('player 1 meeples', state.lib_state.meeples[0]) # type: ignore
                    print('player 2 meeples', state.lib_state.meeples[1]) # type: ignore
                    if not state.is_terminal():
                        print('allowed actions: ', state.get_legal_actions())
                    state.visualize()
                    input()

                if state.is_terminal():
                    break

                action_probs = players[state.get_current_player()].get_action_probs(state, temperature=0, add_exploration_noise=False)
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                action = random.choices(actions, weights=probs)[0]
                if debug.arena:
                    print("[debug] carcassonne_train.py:107 c9de62 action = {}".format(action))

                state = state.apply_action(action)

            # return winner id, or -1 for draw
            values = [state.get_player_value(i) for i in range(state.get_num_players())]
            if debug.arena: print("[debug] carcassonne_train.py:101 9cfb5d values = {}".format(values))
            max_value = max(values)
            if debug.arena: print("[debug] carcassonne_train.py:103 b2092b max_value = {}".format(max_value))
            winners = [i for i, v in enumerate(values) if v == max_value]
            if debug.arena: print("[debug] carcassonne_train.py:105 b2f528 winners = {}".format(winners))
            winner = winners[0] if len(winners) == 1 else -1
            if debug.arena: print("[debug] carcassonne_train.py:107 5320d8 winner = {}".format(winner))
            scores = [state.get_player_score(i) for i in range(state.get_num_players())]
            return winner, scores

        score1_sum = 0
        score2_sum = 0
            
        for _ in tqdm(range(iterations), desc="Arena.playGames (1)", mininterval=1, leave=False):
            result, scores = play_game([agent1, agent2])
            if result == 0:
                win1 += 1
            elif result == 1:
                win2 += 1
            else:
                draws += 1
            score1_sum += scores[0]
            score2_sum += scores[1]

        for _ in tqdm(range(iterations), desc="Arena.playGames (2)", mininterval=1, leave=False):
            result, scores = play_game([agent2, agent1])
            if result == 1:
                win1 += 1
            elif result == 0:
                win2 += 1
            else:
                draws += 1
            score1_sum += scores[1]
            score2_sum += scores[0]

        score1 = score1_sum / (iterations * 2)
        score2 = score2_sum / (iterations * 2)

        return win1, win2, draws, score1, score2

    def save():
        path = checkpoints_manager.next_checkpoint_path(current_iteration)
        best_model_path = os.path.join(path, 'best')
        os.makedirs(best_model_path, exist_ok=True)
        agent.save(best_model_path)
        training_state_path = os.path.join(path, 'training_state.json')
        with open(training_state_path, 'w') as f:
            json.dump({
                'current_iteration': current_iteration,
            }, f)
        training_data_path = os.path.join(path, 'training_data.json')
        with open(training_data_path, 'w') as f:
            json.dump(train_data_by_iter_to_json(train_data_by_iter), f)

    def load(path):
        nonlocal current_iteration_start, train_data_by_iter
        best_model_path = os.path.join(path, 'best')
        agent.load(best_model_path)
        training_state_path = os.path.join(path, 'training_state.json')
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
            current_iteration_start = training_state['current_iteration'] + 1

        train_data_path = os.path.join(path, 'training_data.json')
        with open(train_data_path, 'r') as f:
            print('Loading training data...')
            train_data_by_iter = train_data_by_iter_from_json(json.load(f))

    if load_checkpoint_path is not None:
        print('Loading...')
        load(load_checkpoint_path)

    for current_iteration in tqdm(range(current_iteration_start, current_iteration_start + num_iters), desc="Iterations"):
        # Execute self play and collect data
        current_train_data = deque([], maxlen=max_train_data_per_iter)

        for i in tqdm(range(num_self_play_episodes), desc="Self play"):
            current_train_data += execute_episode(current_iteration * num_self_play_episodes + i)

        # save the iteration examples to the history 
        train_data_by_iter.append(current_train_data)

        while len(train_data_by_iter) > max_train_data_iters:
            train_data_by_iter.pop(0)

        # Train the agent
        all_train_data = []
        for train_data in train_data_by_iter:
            all_train_data.extend(train_data)

        agent.save(temp_save_path)
        prev_agent.load(temp_save_path)

        agent.train(all_train_data, current_iteration=current_iteration)

        if arena_win_threshold > 0:
            print('New model against previous:')
            prev_win, new_win, prev_new_draw, prev_score, new_score = play_arena_games(game, arena_iterations, prev_agent, agent)
            print(f'prev win: {prev_win}, new win: {new_win}, prev_new_draw: {prev_new_draw}')
            
            with ex.summary_writer.as_default():
                tf.summary.scalar('win_rate_against_prev', float(new_win) / arena_iterations, step=current_iteration)
                tf.summary.scalar('draw_rate_against_prev', float(prev_new_draw) / arena_iterations, step=current_iteration)
                tf.summary.scalar('prev_score', prev_score, step=current_iteration)
                tf.summary.scalar('new_score', new_score, step=current_iteration)

            if prev_win + new_win == 0 or float(new_win) / (prev_win + new_win) < arena_win_threshold:
                print('Rejecting new model')
                agent.load(temp_save_path)
            else:
                print('Accepting new model')

        save()

        print('Current model against random:')
        agent_win, random_win, agent_random_draw, agent_score, random_score = play_arena_games(game, random_arena_iterations, agent, random_agent)
        print(f'agent win: {agent_win}, random win: {random_win}, agent_random_draw: {agent_random_draw}')

        with ex.summary_writer.as_default():
            tf.summary.scalar('win_rate_against_random', float(agent_win) / random_arena_iterations, step=current_iteration)
            tf.summary.scalar('draw_rate_against_random', float(agent_random_draw) / random_arena_iterations, step=current_iteration)
            tf.summary.scalar('agent_score', agent_score, step=current_iteration)
            tf.summary.scalar('random_score', random_score, step=current_iteration)