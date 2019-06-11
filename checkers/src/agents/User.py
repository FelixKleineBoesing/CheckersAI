import numpy as np

from checkers.src.agents.Agent import Agent
from checkers.src.Helpers import ActionSpace


class User(Agent):
    """
    This is only a placeholder for user input through rest interface
    """
    def decision(self, state_space: np.ndarray, action_space: ActionSpace):
        pass
