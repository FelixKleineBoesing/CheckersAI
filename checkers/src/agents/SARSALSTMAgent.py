import tensorflow as tf
from keras.layers import Dense, Flatten, LSTM
import keras
import numpy as np
import os

from checkers.src.ReplayBuffer import EpisodeBuffer
from checkers.src.agents.SARSAAgent import SARSAAgent
from checkers.src.agents.Agent import Agent


class SARSALSTMAgent(SARSAAgent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5,
                 intervall_turns_train: int = 500, intervall_turns_load: int = 10000,
                 saver_path: str = "../data/modeldata/sarsalstm/model.ckpt"):
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
        self.sess = tf.Session()

        # calculate number actions from actionshape
        self.number_actions = np.product(action_shape)
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        # buffer variables
        self._buffer_action = None
        self._buffer_state = None
        self._buffer_reward = None
        self._buffer_done = None

        self.target_network = self._configure_network(state_shape, "target_{}".format(name))
        self.network = self._configure_network(state_shape, name)
        self._batch_size = 2048

        # prepare a graph for agent step
        self.state_t = tf.placeholder('float32', [None, ] + list((1, state_shape[0] * state_shape[1])))
        self.qvalues_t = self._get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_{}".format(name))

        # init placeholder
        self._obs_ph = tf.placeholder(tf.float32, shape=(None,) + (1, state_shape[0] * state_shape[1]))
        self._actions_ph = tf.placeholder(tf.int32, shape=[None])
        self._rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self._next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + (1, state_shape[0] * state_shape[1]))
        self._is_done_ph = tf.placeholder(tf.float32, shape=[None])
        self._next_actions_ph = tf.placeholder(tf.int32, shape=[None])

        self.saver = tf.train.Saver()
        self._saver_path = saver_path
        self._configure_target_model()
        if os.path.isfile(self._saver_path + ".index"):
            self.saver.restore(self.sess, self._saver_path)

        # copy weight to target weights
        self.load_weigths_into_target_network()
        self.exp_buffer = EpisodeBuffer(5000)
        self.side = side
        self.state_shape = state_shape
        self.action_shape = action_shape
        self._number_turns = 0

    def _configure_network(self, state_shape: tuple, name: str):
        # define network
        with tf.variable_scope(name, reuse=False):
            network = keras.models.Sequential()
            network.add(LSTM(64, activation="relu", input_shape=(1, 64), return_sequences=True))
            network.add(LSTM(128, activation="relu", return_sequences=True))
            network.add(LSTM(256, activation="relu", return_sequences=True))
            network.add(Dense(512, activation="relu"))
            network.add(Dense(512, activation="relu"))
            network.add(Dense(512, activation="relu"))
            network.add(Flatten())
            network.add(Dense(self.number_actions, activation="linear"))
        return network

    def _get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.network(state_t)
        return qvalues

    def _get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        return self.sess.run(self.qvalues_t, {self.state_t:
                                                  state_t[0].reshape(1, 1, state_t[0].shape[0] * state_t[0].shape[1])})

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, next_act_batch = \
            self.exp_buffer.sample(batch_size)
        obs_batch = obs_batch.reshape(obs_batch.shape[0], 1, obs_batch.shape[1] * obs_batch.shape[2])
        next_obs_batch = next_obs_batch.reshape(next_obs_batch.shape[0], 1, next_obs_batch.shape[1] * next_obs_batch.shape[2])
        return {
            self._obs_ph: obs_batch, self._actions_ph: act_batch, self._rewards_ph: reward_batch,
            self._next_obs_ph: next_obs_batch, self._is_done_ph: is_done_batch, self._next_actions_ph: next_act_batch
        }