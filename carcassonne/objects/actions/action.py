action_to_idx_fn = None


class Action:
    def __init__(self):
        self._idx = None

    def _get_idx(self):
        global action_to_idx_fn
        if self._idx is None:
            if action_to_idx_fn is None:
                from agents.carcassonne_agent import action_to_idx
                action_to_idx_fn = action_to_idx

            self._idx = action_to_idx_fn(self)

        return self._idx

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Action):
            return False

        return self._get_idx() == __o._get_idx()

    def __hash__(self) -> int:
        return self._get_idx()
