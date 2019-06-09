from typing import Union
import numpy as np
import pdb
import simplejson
import os
import logging


class Rewards:

    def __init__(self, win: Union[float, int],
                 loss: Union[float, int],
                 queen_stone_loss: Union[float, int],
                 normal_stone_loss: Union[float, int],
                 normal_stone_taken: Union[float, int],
                 queen_stone_taken: Union[float, int],
                 turn: Union[float, int]):
        """
        This class represents the reward function which returns a reward for a discrete set of actions
        :param win: reward for winning a game
        :param queen_stone_loss: reward for loosing a queen stone
        :param normal_stone_loss: reward for loosing a normal stone
        :param normal_stone_taken: reward for taking a normal stone
        :param queen_stone_taken: reward for taking a queen stone
        :param turn: reward for playing a turn
        """
        assert all(isinstance(arg, float) or isinstance(arg, int) for arg in [win, loss, queen_stone_loss,
                                                                              queen_stone_taken, normal_stone_loss,
                                                      normal_stone_taken])

        self.win = win
        self.loss = loss
        self.normal_stone_loss = normal_stone_loss
        self.queen_stone_loss = queen_stone_loss
        self.normal_stone_taken = normal_stone_taken
        self.queen_stone_taken = queen_stone_taken
        self.turn = turn


default_rewards = Rewards(5000, -5000, 0, 0, 10, 10, 0)


class ActionSpace:

    def __init__(self, board_length: int = 8):
        """
        class that represents the action space and stores the action space as a dict and an array
        The array just stores the possibles moves with a 4D array (X From, Y from, X To, Y To). possible  move has
        value 1, otherwise zero
        The dictionaray  store informations (specific stone id, the enemy stones, that would be jumped over and the
        value of those stones
        :param board_length length of board sides
        """
        self.space_dict = {}
        self.space_array = np.full([board_length] * 4, np.nan)

    def update_dict(self, key, value):
        # update dict first
        if key not in self.space_dict:
            if type(value) != list:
                value = [value]
            self.space_dict[key] = value
        else:
            self.space_dict[key].append(value)

    def update_array(self, coord: np.ndarray, new_coord: np.ndarray):
        # update array
        self.space_array[coord[0], coord[1], new_coord[0], new_coord[1]] = 1

    def keys(self):
        return self.space_dict.keys()

    def __len__(self):
        return len(self.space_dict)

    def __getitem__(self, item):
        try:
            return self.space_dict[item]
        except KeyError:
            pdb.set_trace()


class Config:

    def __init__(self):
        self._load_from_file()

    def _load_from_file(self):
        with open("../../config.json", "r") as f:
            self._store = simplejson.load(f)

    def _get_from_env(self, key: str):
        return os.environ[key]

    def __getitem__(self, item):
        if os.path.isfile("../../is_docker"):
            try:
                return self._get_from_env(item)
            except KeyError:
                logging.error("Key {} is not present in environment variables!")
        else:
            try:
                return self._store[item]
            except KeyError:
                logging.error("Key {} is not present in config.json!")


def multiply(*args):
    """
    Helper function which multiplies the all numbers that are delivered in the function call
    :param args: numbers
    :return: product of all numbers
    """
    product = 1
    for arg in args:
        product *= arg
    return product


def min_max_scaling(arr: np.ndarray, min_val: float=-2.0, max_val: float=2.0):
    """
    scales the given array between 0 and 1
    :param arr: numpy array
    :param min_val: min occurence in data, default is -2.0 based on the min possible stone value
    :param max_val: max occurence in data, default is +2.0 based on the max possible stone value
    :return: scaled numpy array
    """
    return (arr.astype('float32') - min_val) / (max_val - min_val)
