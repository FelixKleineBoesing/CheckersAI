import abc
import numpy as np


class Agent(abc.ABC):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up"):
        """
        abstract class for agent which define the general interface for Agents
        :param name:
        :param side:
        """
        assert side in ["up", "down"], "Side has to be up or down"
        self.name = name
        self.side = side

    def play_turn(self, state_space: np.ndarray, action_space: dict):
        """
        get all possible actions and decide which action to take
        :param action_space:
        :return:
        """
        decision = self.decision(state_space, action_space)
        move = action_space[decision["stone_id"]][decision["move_id"]]
        stone_id = decision["stone_id"]
        return move, stone_id

    @abc.abstractmethod
    def decision(self, state_space: np.ndarray, action_space: dict):
        """
        this function must implement a decision based in the action_space and other delivered arguments
        return must be a dictionary with the following keys: "stone_id" and "move_index" which indicates
        the stone and move that should be executed
        :param action_space:
        :return:
        """
        pass