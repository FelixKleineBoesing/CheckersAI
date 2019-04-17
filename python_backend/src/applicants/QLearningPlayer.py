from ..Agent import Agent
import random
import numpy as np


class DeepQAgent(Agent):

    def __init__(self, network, name: str, side: str = "up"):
        """
        Agent which implements Q Learning
        :param network:
        :param name:
        :param side:
        """
        self.network = network
        super().__init__(name, side)

    def decision(self, action_space: dict):
        pass

        #return {"stone_id": stone_id, "move_id": move_id}


class Learner:

    def __init__(self, network):
        """
        Learner, which takes the states and actions from Agent and learn with the same network
        :param network:
        """
        self.network = network


    def train_network(self, states: np.ndarray, actions: np.ndarray):
        """
        trains network with states and actions from Agent
        :param states:
        :param actions:
        :return:
        """
        pass

    def return_network_weights(self):
        """
        returns network weights for updating of Agent
        :return:
        """
        pass