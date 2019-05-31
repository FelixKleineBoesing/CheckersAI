import abc
import numpy as np
import logging
import datetime

from checkers.src.Helpers import ActionSpace
from checkers.src.Helpers import Config
from checkers.src.cache.RedisWrapper import RedisCache, RedisChannel
from checkers.src.cache.RedisWrapper import get_key


class Agent(abc.ABC):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up",
                 config: Config = None, caching: bool = False,
                 cache: RedisCache = None, channel: RedisChannel = None):
        """
        abstract class for agent which define the general interface for Agents
        :param name:
        :param side:
        """
        assert side in ["up", "down"], "Side has to be up or down"
        self.name = name
        self.side = side
        self.state_shape = state_shape
        self.action_shape = action_shape
        self._number_turns = 0
        self._td_loss_history = []
        self._moving_average_loss = []
        self._reward_history = []
        self._moving_average_rewards = []
        if config is None:
            config = Config()

        if caching:
            logging.debug("Caching is considered! When you don´t deliver cache and stream by yourself, the agent will "
                          "get a redis stream and cache by default")
            if cache is None:
                self.redis_cache = RedisCache(host=config["host"], port=config["port"], db=int(config["db"]))
            else:
                self.redis_cache = cache
            if channel is None:
                self.redis_channel = RedisChannel(host=config["host"], port=config["port"], db=int(config["db"]))
            else:
                self.redis_channel = channel
        else:
            self.redis_cache = None
            self.redis_channel = None
            logging.info("No caching is considered!")

    def play_turn(self, state_space: np.ndarray, action_space: ActionSpace):
        """
        get all possible actions and decide which action to take
        :param state_space: np array describing the board
        :param action_space: dictionary containing all possible moves
        :return:
        """
        decision = self.decision(state_space, action_space)
        assert isinstance(decision, np.ndarray), "decision return must be a numpy array"
        assert len(decision) == 4, "decision return must be a np array with length 4"
        self._number_turns += 1
        return decision

    def get_feedback(self, state, action, reward, next_state, finished):
        """
        through this function the agent gets information about the last turn
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param finished:
        :return: No return
        """
        pass

    @abc.abstractmethod
    def decision(self, state_space: np.ndarray, action_space: ActionSpace):
        """
        this function must implement a decision based in the action_space and other delivered arguments
        return must be a dictionary with the following keys: "stone_id" and "move_index" which indicates
        the stone and move that should be executed
        :param action_space:
        :return: np.array(X_From, Y_From, X_To, Y_To)
        """
        pass

    def publish_data(self):
        data = {"rewards": self._reward_history,
                "avg_rewards": self._moving_average_rewards,
                "loss": self._td_loss_history,
                "avg_loss": self._moving_average_loss}
        key = get_key(self.name)
        self._put_in_cache(key, data)
        self._put_in_channel(self.name, key)

    def _put_in_cache(self, key: str, data: dict):
        self.redis_cache.put_data_into_cache(key, data)

    def _put_in_channel(self, channel_name: str, key: str):
        self.redis_channel.put_into_channel(channel_name, {"target": "stats", "data_key": key})