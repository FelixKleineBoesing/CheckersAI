import random
import numpy as np

<<<<<<< HEAD:checkers/agents/random_agent.py
from checkers.src.agents.agent import Agent
from checkers.src.game.GameHelpers import ActionSpace
=======
from checkers.agents.Agent import Agent
from checkers.game.GameHelpers import ActionSpace
>>>>>>> 24a1d78fa109a2a5e99bdba05fb2c26a86a3f7f1:checkers/agents/RandomAgent.py


class RandomAgent(Agent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up"):
        """
        Agent which chooses actions randomly
        :param name:
        :param side:
        """
        super().__init__(state_shape, action_shape, name, side)

    def decision(self, state_space: np.ndarray, action_space: ActionSpace):
        """
        choose action randomly from given actions space
        :param state_space:
        :param action_space:
        :return:
        """
        keys = [key for key in action_space.keys()]
        ix = random.sample(range(len(keys)), 1)[0]
        stone_id = keys[ix]
        move_id = random.sample(range(len(action_space[stone_id])), 1)[0]
        move = action_space[stone_id][move_id]
        decision = np.concatenate([move["old_coord"], move["new_coord"]])

        return decision
