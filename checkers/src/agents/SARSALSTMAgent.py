import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, LSTM
import keras
import numpy as np
import os

from checkers.src.Helpers import Config, min_max_scaling
from checkers.src.ReplayBuffer import EpisodeBuffer
from checkers.src.agents.SARSAAgent import SARSAAgent
from checkers.src.agents.Agent import Agent
from checkers.src.cache.RedisWrapper import RedisChannel, RedisCache


class SARSALSTMAgent(SARSAAgent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5,
                 intervall_turns_train: int = 500, intervall_turns_load: int = 10000,
                 save_path: str = "../data/modeldata/sarsalstm/model.ckpt", caching: bool = False,
                 config: Config = None, cache: RedisCache = None, channel: RedisChannel = None):
        """
        Agent which implements Q Learning
        :param state_shape: shape of state
        :param action_shape: shape of actions
        :param name: name of agent
        :param side: is he going to start at the top or at the bottom?
        :param epsilon: exploration factor
        """
        # TODO reshape states as lstm expects 3d instead of 2d arrays
        # tensorflow related stuff
        self.name = name
        self._batch_size = 512
        self._learning_rate = 0.01

        # calculate number actions from actionshape
        self.number_actions = np.product(action_shape)
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        # buffer variables
        self._buffer_action = None
        self._buffer_state = None
        self._buffer_reward = None
        self._buffer_done = None

        self.target_network = self._configure_network(state_shape)
        self.network = self._configure_network(state_shape)

        self.epsilon = epsilon

        self._save_path = save_path
        if os.path.isfile(self._save_path + ".index"):
            self.network.load_weights(self._save_path)

        # copy weight to target weights
        self.load_weights_into_target_network()
        self.exp_buffer = EpisodeBuffer(5000)
        self.side = side
        self.state_shape = state_shape
        self.action_shape = action_shape
        self._number_turns = 0
        Agent.__init__(self, state_shape, action_shape, name, side, config, caching, cache, channel)

    def _configure_network(self, state_shape: tuple):
        network = tf.keras.models.Sequential([
            LSTM(512, activation="relu", input_shape=(1, 64), return_sequences=True),
            LSTM(1024, activation="relu", return_sequences=True),
            #LSTM(512, activation="relu", return_sequences=True),
            #LSTM(1024, activation="relu", return_sequences=True),
            #Dense(1024, activation="relu"),
            #Dense(2048, activation="relu"),
            Dense(1024, activation="relu"),
            Flatten(),
            Dense(self.number_actions, activation="linear")])
        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        return network

    def _get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        return self.network(state_t[0].reshape(1, 1, state_t[0].shape[0] * state_t[0].shape[1]))

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, next_act_batch = \
            self.exp_buffer.sample(batch_size)
        obs_batch = obs_batch.reshape(obs_batch.shape[0], 1, obs_batch.shape[1] * obs_batch.shape[2])
        next_obs_batch = next_obs_batch.reshape(next_obs_batch.shape[0], 1, next_obs_batch.shape[1] * next_obs_batch.shape[2])
        obs_batch = min_max_scaling(obs_batch)
        next_obs_batch = min_max_scaling(next_obs_batch)
        is_done_batch = is_done_batch.astype("float32")
        reward_batch = reward_batch.astype("float32")
        return {"obs": obs_batch, "actions": act_batch, "rewards": reward_batch,
                "next_obs": next_obs_batch, "is_done": is_done_batch , "next_actions": next_act_batch}
