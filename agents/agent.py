from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar, Generic

from games.game_state import GameState

TState = TypeVar('TState', bound=GameState)

class Agent(ABC, Generic[TState]):
    def reset_inference(self) -> None:
        pass

    @abstractmethod
    def get_action_probs(self, state: TState, temperature: float, add_exploration_noise: bool) -> Dict[Any, float]:
        pass