import numpy as np
import simplejson
import os
import logging


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


def update_managed_dict(managed_dict, game_id, key, value):
    content = managed_dict[game_id]
    content[key] = value
    managed_dict[game_id] = content