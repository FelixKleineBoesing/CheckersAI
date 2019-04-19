from ..Agent import Agent
import random
import numpy as np

class RandomAgent(Agent):

    def __init__(self, name: str, side: str = "up"):
        """
        Agent which chooses actions randomly
        :param name:
        :param side:
        """
        super().__init__(name, side)

    def decision(self, state_space: np.ndarray, action_space: dict):
        """
        choose action randomly from given actions space
        :param action_space:
        :return:
        """
        keys = [key for key in action_space.keys()]
        ix = random.sample(range(len(keys)), 1)[0]
        stone_id = keys[ix]

        move_id = random.sample(range(len(action_space[stone_id])), 1)[0]

        return {"stone_id": stone_id, "move_id": move_id}
