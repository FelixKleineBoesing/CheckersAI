import abc
import numpy as np
import logging

from checkers.src.Helpers import ActionSpace
from checkers.src.Helpers import Config
from checkers.src.cache.RedisWrapper import RedisCache, RedisChannel


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
        if caching:
            logging.debug("Caching is considered! When you don´t deliver cache and stream by yourself, the agent will "
                          "get a redis stream and cache by default")
            if cache is None:
                self.redis_cache = RedisCache()
            else:
                self.redis_cache = cache
            if channel is None:
                self.redis_stream = RedisChannel()
            else:
                self.redis_stream = channel
        else:
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

    def publish_data(self, key, value):
        pass

    def _put_in_cache(self):
        pass

    def _put_in_stream(self):
        pass