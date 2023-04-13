from games.game_state import GameState
from .agent import Agent

class RandomAgent(Agent):
    def get_action_probs(self, state: GameState, temperature: float, add_exploration_noise: bool):
        action_probs = {}
        legal_actions = state.get_legal_actions()
        if len(legal_actions) == 0:
            return action_probs

        prob = 1.0 / len(legal_actions)
        for action in legal_actions:
            action_probs[action] = prob

        return action_probs