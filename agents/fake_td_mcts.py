
import numpy as np
from games.game_state import GameState
from utils.utils import normalize_dict


class FakeTDMCTS(object):
    def __init__(self, model) -> None:
        self.model = model

    def get_action_probs(self, state: GameState, temperature=1.0, add_exploration_noise=False):
        all_legal_actions = state.get_legal_actions()
        assert len(all_legal_actions) > 0

        best_action = None
        best_value = -np.inf
        for action in all_legal_actions:
            child_state = state.apply_action(action)
            model_output = self.model(child_state, child_state.get_legal_actions())
            pred_value = model_output['value']
            if state.get_current_player() != child_state.get_current_player():
                pred_value = -pred_value

            if pred_value > best_value:
                best_action = action
                best_value = pred_value

        policy = {
            action: 1.0 if action == best_action else 0.0
            for action in all_legal_actions
        }

        # Add exploration noise for root
        if add_exploration_noise:
            # noise = np.random.dirichlet([0.25] * len(all_legal_actions))
            # for a, n in zip(all_legal_actions, noise):
            #     policy[a] = policy[a] * (1 - 0.25) + n * 0.25
            pass

        for a in policy:
            policy[a] = policy[a] + temperature * (0.5 / len(all_legal_actions))

        policy = normalize_dict(policy)
        assert policy is not None
        return policy