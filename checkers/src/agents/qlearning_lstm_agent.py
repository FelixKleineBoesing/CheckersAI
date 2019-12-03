import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, LSTM
import numpy as np

from checkers.src.agents.qlearning_agent import QLearningAgent
from checkers.src.ReplayBuffer import EpisodeBuffer
from checkers.src.cache.RedisWrapper import RedisChannel, RedisCache
from checkers.src.Helpers import Config, multiply


class QLearningLSTMAgent(QLearningAgent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.0,
                 intervall_turns_train: int = np.Inf, intervall_turns_load: int = np.Inf,
                 save_path: str = "../data/modeldata/qlstm/model.ckpt", caching: bool = False,
                 config: Config = None, cache: RedisCache = None, channel: RedisChannel = None):
        """
        Agent which implements Q Learning
        :param state_shape: shape of state
        :param action_shape: shape of actions
        :param name: name of agent
        :param side: is he going to start at the top or at the bottom?
        :param epsilon: exploration factor
        """
        super().__init__(state_shape, action_shape, name, side, epsilon, intervall_turns_train, intervall_turns_load,
                         save_path, caching, config, cache, channel)
        self.exp_buffer = EpisodeBuffer(5000)

    def _configure_network(self, state_shape: tuple):
        network = tf.keras.models.Sequential([
            LSTM(512, activation="relu", input_shape=(1, multiply(*state_shape)), return_sequences=True),
            #LSTM(1024, activation="relu", return_sequences=True),
            #LSTM(2048, activation="relu", return_sequences=True),
            #LSTM(4096, activation="relu", return_sequences=True),
            #Dense(8012, activation="relu"),
            #Dense(4096, activation="relu"),
            Dense(2048, activation="relu"),
            Flatten(),
            Dense(self.number_actions, activation="linear")])
        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        return network
