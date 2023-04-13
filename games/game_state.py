from abc import ABC, abstractmethod
from typing import Any

class GameState(ABC):
    @abstractmethod
    def get_legal_actions(self) -> Any:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def get_num_players(self) -> int:
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        pass

    @abstractmethod
    def get_player_value(self, player: int) -> float:
        pass

    @abstractmethod
    def get_player_score(self, player: int) -> float:
        """
        Returns a human-readable score of the given player. This is not necessarily the same as the value.
        """
        pass
        
    @abstractmethod
    def apply_action(self, action: Any) -> 'GameState':
        pass

    @abstractmethod
    def visualize(self) -> None:
        pass

    @abstractmethod
    def get_reuse_hash_key(self) -> Any:
        """
        Returns a hashable object that can be used to determine if two states are the same.
        """
        pass