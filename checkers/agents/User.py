import numpy as np

<<<<<<< HEAD:checkers/src/agents/User.py
from checkers.src.agents.agent import Agent
from checkers.src.game.GameHelpers import ActionSpace
=======
from checkers.agents.Agent import Agent
from checkers.game.GameHelpers import ActionSpace
>>>>>>> 24a1d78fa109a2a5e99bdba05fb2c26a86a3f7f1:checkers/agents/User.py


class User(Agent):
    """
    This is only a placeholder for user input through rest interface
    """
    def decision(self, state_space: np.ndarray, action_space: ActionSpace):
        pass
