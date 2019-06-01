import tensorflow as tf
from keras.layers import Dense, Flatten, LSTM, Dropout, Input
import keras
import numpy as np
import os
import logging

from checkers.src.ReplayBuffer import ReplayBuffer
from checkers.src.agents.Agent import Agent


class A3C(Agent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5,
                 intervall_turns_train: int = 500, intervall_turns_load: int = 10000,
                 saver_path: str = "../data/modeldata/a3c/model.ckpt"):
        #TODO finalize
        # tensorflow related stuff
        self.name = name
        self.sess = tf.Session()

        # calculate number actions from actionshape
        self.number_actions = np.product(action_shape)
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        self.target_network = self._configure_network(state_shape, "target_{}".format(name))
        self.network = self._configure_network(state_shape, self.name)

        # prepare a graph for agent step
        self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
        self.qvalues_t = self._get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_{}".format(name))
        self.exp_buffer = ReplayBuffer(100000)
        self._batch_size = 4096

        # init placeholder
        self._obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_shape)
        self._actions_ph = tf.placeholder(tf.int32, shape=[None])
        self._rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self._next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_shape)
        self._is_done_ph = tf.placeholder(tf.float32, shape=[None])

        self.saver = tf.train.Saver()
        self._saver_path = saver_path
        self._configure_target_model()
        if os.path.isfile(self._saver_path + ".index"):
            self.saver.restore(self.sess, self._saver_path)

        # copy weight to target weights
        self.load_weigths_into_target_network()

        super().__init__(state_shape, action_shape, name, side)

    def _configure_network(self, state_shape: tuple, name: str):
        # define network
        with tf.variable_scope(name, reuse=False):
            inputs = Input(shape=(1, 64))
            x = LSTM(128, activation="relu", input_shape=(1, 64), return_sequences=True)(inputs)
            x = LSTM(256, activation="relu", return_sequences=True)(x)
            x = LSTM(512, activation="relu", return_sequences=True)(x)
            x = Dense(1024, activation="relu")(x)
            x = Dense(2048, activation="relu")(x)
            x = Dense(1024, activation="relu")(x)
            x = Flatten()(x)

            logits = Dense(self.number_actions, activation="linear")(x)
            state_value = Dense(1, activation="linear")(x)
            network = keras.models.Model(inputs=inputs, outputs=[logits, state_value])

            self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
            self.agent_outputs = self._get_symbolic_qvalues(self.state_t)

        return network

    def load_weigths_into_target_network(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        logging.debug("Transfer Weight!")
        assigns = []
        for w_agent, w_target in zip(self.weights, self.target_weights):
            assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
        self.sess.run(assigns)
        self.saver.save(self.sess, self._saver_path)

    def train_network(self):
        logging.debug("Train Network!")
        _, loss_t = self.sess.run([self._train_step, self._entropy], self._sample_batch(batch_size=self._batch_size))
        self._td_loss_history.append(loss_t)
        self._moving_average_loss.append(np.mean([self._td_loss_history[max([0, len(self._td_loss_history) - 100]):]]))
        ma = self._moving_average_loss[-1]
        relative_ma = self._moving_average_loss[-1] / self._batch_size
        logging.info("Loss: {},     relative Loss: {}".format(ma, relative_ma))

    def _get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues, state_values = self.network(state_t)
        return qvalues, state_values

    def _get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        return self.sess.run(self.qvalues_t, {self.state_t:
                                                  state_t[0].reshape(1, 1, state_t[0].shape[0] * state_t[0].shape[1])})

    def _configure_target_model(self):
        # placeholders that will be fed with exp_replay.sample(batch_size)
        is_not_done = 1 - self._is_done_ph
        gamma = 0.99
        qvalues, state_values = self._get_symbolic_qvalues(self._obs_ph)

        next_qvalues, next_state_values = self.target_network(self._next_obs_ph)
        next_state_values = next_state_values * is_not_done
        probs = tf.nn.softmax(qvalues)
        logprobs = tf.nn.log_softmax(qvalues)

        logp_actions = tf.reduce_sum(logprobs * tf.one_hot(self._actions_ph, self.number_actions), axis=-1)
        advantage = self._rewards_ph + gamma * next_state_values - state_values
        self._entropy = -tf.reduce_sum(probs * logprobs, 1, name="entropy")
        self._actor_loss = - tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - 0.001 * tf.reduce_mean(entropy)
        target_state_values = self._rewards_ph + gamma * next_state_values
        self._critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values)) ** 2)

        self._train_step = tf.train.AdamOptimizer(1e-3).minimize(self._actor_loss + self._critic_loss )
        self.sess.run(tf.global_variables_initializer())

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = self.exp_buffer.sample(batch_size)
        return {
            self._obs_ph: obs_batch, self._actions_ph: act_batch, self._rewards_ph: reward_batch,
            self._next_obs_ph: next_obs_batch, self._is_done_ph: is_done_batch
        }