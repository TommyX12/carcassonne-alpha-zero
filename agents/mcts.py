from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Any, Dict, Optional

import numpy as np

from utils.utils import argmax_dict, keydefaultdict, normalize_dict
from utils.debug import debug

from games.game_state import GameState


@dataclass
class MCTSConfig:
    num_search: int = 25
    cpuct: float = 1.25
    root_dirichlet_alpha: float = 0.25
    root_exploration_fraction: float = 0.05
    enable_exploration_noise: bool = True
    enable_value_normalization: bool = False


class Node(object):
    def __init__(self, parent: Optional['Node']=None, parent_action=None, prior=0) -> None:
        self.parent = parent
        self.parent_action = parent_action
        self.state: GameState = None  # type: ignore
        # visit count
        self.visit_count = 0
        # value estimate
        self.value_estimate = 0
        self.expanded = False
        # parent predicted policy p(s, a)
        self.prior = prior
        # children, mapping from action to node
        # guaranteed to be all the legal actions
        self.children = keydefaultdict(lambda action: Node(self, action, 0))

    def expand(self, state, policy, value):
        self.state = state
        self.value_estimate = value
        self.expanded = True
        self.children = {
            action: Node(self, action, prob)
            for action, prob in policy.items()
        }


class MCTS(object):
    def __init__(self, model, config: Optional[MCTSConfig]=None) -> None:
        super().__init__()

        if config is None:
            config = MCTSConfig()

        self.model = model
        self.config = config
        self.reset()
        self.root = Node()
        self.root_exploration_noise: Optional[Dict[Any, float]] = None

    def reset(self):
        self.current_min = float('inf')
        self.current_max = float('-inf')
        self.all_node_map: Dict[Any, Node] = {}

    def _update_min_max(self, node: Node):
        value = node.value_estimate
        if node.parent and node.parent.state.get_current_player() != node.state.get_current_player():
            # negate for the other player
            value = -value

        self.current_min = min(self.current_min, value)
        self.current_max = max(self.current_max, value)

    def _get_selection_score(self, parent_node: Node, child_node: Node):
        # We already increased the visit count of the parent node by the time we call this function
        value_estimate = 0.0 # default value for unexpanded nodes
        if child_node.expanded:
            value_estimate = child_node.value_estimate

            # Negate value if needed
            if parent_node.state.get_current_player() != child_node.state.get_current_player():
                value_estimate = -value_estimate

            if self.config.enable_value_normalization:
                if self.current_max > self.current_min:
                    value_estimate = (value_estimate - self.current_min) / (self.current_max - self.current_min)

                else:
                    value_estimate = 0.0

        prior = child_node.prior
        # Apply dirichlet noise for the root node
        if parent_node == self.root and self.root_exploration_noise is not None and self.config.enable_exploration_noise:
            prior = prior * (1 - self.config.root_exploration_fraction) + self.root_exploration_noise[child_node.parent_action] * self.config.root_exploration_fraction

        parent_old_count = parent_node.visit_count - 1
        return value_estimate + self.config.cpuct * child_node.prior * math.sqrt(parent_old_count) / (1 + child_node.visit_count)

    def _search(self, node: Node) -> float:
        """
        Run one iteration of MCTS.
        Returns the sample value, for back-propagation.
        """
        node.visit_count += 1

        if debug: print("[debug] mcts.py:79 a2af89 _search on node, ", node)
        if debug: print("[debug] mcts.py:80 37281e node.prior = {}".format(node.prior))
        if debug: print("[debug] mcts.py:81 15e96b node.visit_count = {}".format(node.visit_count))
        if debug: print("[debug] mcts.py:82 24cb86 node.value_estimate = {}".format(node.value_estimate))

        if node.expanded:
            if debug: print("[debug] mcts.py:83 b1f079 node expanded.")
            if node.state.is_terminal():
                if debug: print("[debug] mcts.py:85 e947c3 node is terminal.")
                if debug: print("[debug] mcts.py:86 48a66d node.value_estimate = {}".format(node.value_estimate))
                return node.value_estimate  # guaranteed to be the state's terminal value

            if debug: print("[debug] mcts.py:91 358cdf self.current_min = {}".format(self.current_min))
            if debug: print("[debug] mcts.py:91 358cdf self.current_max = {}".format(self.current_max))
            scores = {
                action: self._get_selection_score(node, child_node)
                for action, child_node in node.children.items()
            }
            assert len(scores) > 0
            if debug: print("[debug] mcts.py:89 c142ea scores = {}".format(scores))
            best_action = argmax_dict(scores)
            if debug: print("[debug] mcts.py:96 ef83ef best_action = {}".format(best_action))
            child_node = node.children[best_action]
            child_sample_return = self._search(child_node)
            if child_node.state.get_current_player() != node.state.get_current_player():
                # negate for the other player
                child_sample_return = -child_sample_return

            if debug: print("[debug] mcts.py:99 6126c8 child sample return = ", child_sample_return)
            if debug: print("[debug] mcts.py:104 c90659 before:")
            if debug: print("[debug] mcts.py:104 366b7d node.value_estimate = {}".format(node.value_estimate))

            # back-propagate
            old_count = node.visit_count - 1
            if old_count == 0:
                node.value_estimate = child_sample_return

            else:
                new_count = node.visit_count
                node.value_estimate = (
                    old_count * node.value_estimate + child_sample_return) / new_count

            if debug: print("[debug] mcts.py:117 d4596c after:")
            if debug: print("[debug] mcts.py:104 366b7d node.value_estimate = {}".format(node.value_estimate))

            self._update_min_max(node)

            if debug: print("[debug] mcts.py:117 58c277 return child_sample_return = ", child_sample_return)

            return child_sample_return

        else:
            if debug: print("[debug] mcts.py:127 929e93 node not expanded. is leaf")
            # leaf node. expand
            if node.parent is None:
                # root node
                state = node.state

            else:
                # non-root node
                state: GameState = node.parent.state.apply_action(node.parent_action)

            if debug: state.visualize()

            sample_return = 0
            if state.is_terminal():
                if debug: print("[debug] mcts.py:142 19d4ea state is terminal.")
                value = state.get_player_value(state.get_current_player())
                node.expand(state, {}, value)
                sample_return = value
            else:
                # Will pre-process input so that it is canonicalized on current player.
                all_legal_actions = state.get_legal_actions()
                if debug: print("[debug] mcts.py:149 2dffb5 all_legal_actions = {}".format(all_legal_actions))
                assert len(all_legal_actions) > 0
                # Output guaranteed to contain all legal actions, and normalized.
                model_output = self.model(state, all_legal_actions)
                pred_policy = model_output['policy']
                pred_value = model_output['value']
                if debug: print("[debug] mcts.py:154 cfda92 pred_policy = {}".format(pred_policy))
                if debug: print("[debug] mcts.py:155 ca61a8 pred_value = {}".format(pred_value))
                node.expand(state, pred_policy, pred_value)
                sample_return = pred_value

            self._update_min_max(node)

            # Update the node map
            state_hash_key = state.get_reuse_hash_key()
            if state_hash_key is not None:
                self.all_node_map[state_hash_key] = node

            return sample_return

    def get_action_probs(self, state: GameState, temperature=1.0, add_exploration_noise=False) -> Dict[Any, float]:
        """
        Given a state, return the probability of action to take.
        Guaranteed to return a dict with legal actions, but may not be all of them.
        """
        assert state.get_num_players() == 2, 'Only 2-player zero-sum games are supported.'

        self.current_min = float('inf')
        self.current_max = float('-inf')

        all_legal_actions = state.get_legal_actions()
        assert len(all_legal_actions) > 0

        # Find node for reuse
        state_hash_key = state.get_reuse_hash_key()
        if state_hash_key is not None and state_hash_key in self.all_node_map:
            self.root = self.all_node_map[state_hash_key]

        else:
            self.root = Node()
            self.root.state = state # for _search to use

        # Add exploration noise for root
        if add_exploration_noise:
            self.root_exploration_noise = {}
            noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(all_legal_actions))
            for a, n in zip(all_legal_actions, noise):
                self.root_exploration_noise[a] = n

        else:
            self.root_exploration_noise = None

        for i in range(self.config.num_search):
            if debug: print("[debug] mcts.py:181 22a30d new search -----")
            self._search(self.root)

        all_counts = {
            action: self.root.children[action].visit_count
            for action in all_legal_actions
        }

        if temperature == 0:
            best_count = max(all_counts.values())
            best_actions = [
                action for action, count in all_counts.items()
                if count == best_count
            ]
            best_action = random.choice(best_actions)
            return {best_action: 1.0}

        else:
            counts = {
                action: count ** (1.0 / temperature)
                for action, count in all_counts.items()
            }
            return normalize_dict(counts) # type: ignore
