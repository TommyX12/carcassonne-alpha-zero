from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from .game_state import GameState

TState = TypeVar('TState', bound=GameState)

class Game(ABC, Generic[TState]):
    @abstractmethod
    def get_initial_state(self) -> TState:
        pass