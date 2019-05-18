import tensorflow as tf
import numpy as np
import os
import logging

from checkers.src.agents.QLearningAgent import QLearningAgent
from checkers.src.ReplayBuffer import ReplayBufferSarsa


class SARSAAgent(QLearningAgent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5,
                 intervall_turns_train: int = 500, intervall_turns_load: int = 10000):
        """
               Agent which implements Q Learning
               :param state_shape: shape of state
               :param action_shape: shape of actions
               :param name: name of agent
               :param side: is he going to start at the top or at the bottom?
               :param epsilon: exploration factor
               """

        # tensorflow related stuff
        self.name = name
        self.sess = tf.InteractiveSession()

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
        self._batch_size = 2048

        # prepare a graph for agent step
        self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
        self.qvalues_t = self._get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon
        self.target_weights = self.weights
        self.exp_buffer = ReplayBufferSarsa(100000)

        # init placeholder
        self._obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_shape)
        self._actions_ph = tf.placeholder(tf.int32, shape=[None])
        self._rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self._next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_shape)
        self._is_done_ph = tf.placeholder(tf.float32, shape=[None])
        self._next_actions_ph = tf.placeholder(tf.int32, shape=[None])

        self.saver = tf.train.Saver()
        self._configure_target_model()
        if os.path.isfile("../data/modeldata/model_sarsa.ckpt.index"):
            self.saver.restore(self.sess, "../data/modeldata/model_sarsa.ckpt")

        # copy weight to target weights
        self.load_weigths_into_target_network()

        super().__init__(state_shape, action_shape, name, side)

    def get_feedback(self, state, action, reward, next_state, finished):
        if self._buffer_action is not None:
            action_number_buffer = np.unravel_index(np.ravel_multi_index(self._buffer_action, self.action_shape),
                                                    (4096,))[0]
            action_number = np.unravel_index(np.ravel_multi_index(action, self.action_shape), (4096,))[0]
            self.exp_buffer.add(self._buffer_state, action_number_buffer, self._buffer_reward, state,
                                self._buffer_done, action_number)
        if self._number_turns % self._intervall_actions_train == 0 and self._number_turns > 1:
            self.train_network()
        if self._number_turns % self._intervall_turns_load == 0 and self._number_turns > 1:
            self.load_weigths_into_target_network()

        self._buffer_action = action
        self._buffer_state = state
        self._buffer_reward = reward
        self._buffer_done = finished
        # if finished set the buffers toi none since we don´t want to mix episodes with each other
        if finished:
            self._buffer_done, self._buffer_reward, self._buffer_state, self._buffer_action = None, None, None, None

    def _configure_target_model(self):
        # placeholders that will be fed with exp_replay.sample(batch_size)
        is_not_done = 1 - self._is_done_ph
        gamma = 0.99
        current_qvalues = self._get_symbolic_qvalues(self._obs_ph)
        current_action_qvalues = tf.reduce_sum(tf.one_hot(self._actions_ph, self.number_actions) * current_qvalues, axis=1)

        # compute q-values for NEXT states with target network
        next_qvalues_target = self.target_network(self._next_obs_ph)
        next_state_values_target = tf.reduce_sum(tf.one_hot(self._next_actions_ph, self.number_actions) *
                                                 next_qvalues_target, axis=1)

        reference_qvalues = self._rewards_ph + gamma * (next_state_values_target * is_not_done)

        # Define loss function for sgd.
        td_loss = (current_action_qvalues - reference_qvalues) ** 2
        self._td_loss = tf.reduce_mean(td_loss)
        self.td_loss_history = []
        self._moving_average = []
        self._train_step = tf.train.AdamOptimizer(1e-3).minimize(self._td_loss, var_list=self.weights)
        self.sess.run(tf.global_variables_initializer())

    def train_network(self):
        _, loss_t = self.sess.run([self._train_step, self._td_loss], self._sample_batch(batch_size=self._batch_size))
        self.td_loss_history.append(loss_t)
        self._moving_average.append(np.mean([self.td_loss_history[max([0,len(self.td_loss_history) - 100]):]]))
        ma = self._moving_average[-1]
        relative_ma = self._moving_average[-1] / self._batch_size
        logging.info("Loss: {},     relative Loss: {}".format(ma, relative_ma))

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, next_act_batch = self.exp_buffer.sample(batch_size)
        return {
            self._obs_ph: obs_batch, self._actions_ph: act_batch, self._rewards_ph: reward_batch,
            self._next_obs_ph: next_obs_batch, self._is_done_ph: is_done_batch, self._next_actions_ph: next_act_batch
        }