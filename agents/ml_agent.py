from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

from games.game_state import GameState

from .agent import Agent

TState = TypeVar('TState', bound=GameState)

class MLAgent(Agent[TState]):
    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def train(self, train_data, **kwargs):
        pass