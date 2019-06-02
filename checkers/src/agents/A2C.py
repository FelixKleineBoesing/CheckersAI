import tensorflow as tf
from keras.layers import Dense, Flatten, LSTM, Dropout, Input
import keras
import numpy as np
import os
import logging
import random

from checkers.src.Helpers import ActionSpace
from checkers.src.ReplayBuffer import ReplayBuffer
from checkers.src.agents.Agent import Agent


class A2C(Agent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5,
                 intervall_turns_train: int = 500, intervall_turns_load: int = 10000,
                 saver_path: str = "../data/modeldata/a2c/model.ckpt"):
        #TODO finalize
        # tensorflow related stuff
        self.name = name
        self.sess = tf.Session()
        self._batch_size = 4096
        self._learning_rate = 0.3

        # calculate number actions from actionshape
        self.number_actions = np.product(action_shape)
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        self.target_network = self._configure_network(state_shape, "target_{}".format(name))
        self.network = self._configure_network(state_shape, self.name)

        # prepare a graph for agent step
        self.state_t = tf.placeholder('float32', [None, ] + list((1, state_shape[0] * state_shape[1])))
        self.qvalues_t = self._get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_{}".format(name))
        self.exp_buffer = ReplayBuffer(100000)


        # init placeholder
        self._obs_ph = tf.placeholder(tf.float32, shape=(None,) + (1, state_shape[0] * state_shape[1]))
        self._actions_ph = tf.placeholder(tf.int32, shape=[None])
        self._rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self._next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + (1, state_shape[0] * state_shape[1]))
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
            x = LSTM(512, activation="relu", input_shape=(1, 64), return_sequences=True)(inputs)
            #x = LSTM(1024, activation="relu", return_sequences=True)(x)
            #x = LSTM(2048, activation="relu", return_sequences=True)(x)
            #x = Dense(4096, activation="relu")(x)
            x = Dense(2048, activation="relu")(x)
            x = Flatten()(x)

            logits = Dense(self.number_actions, activation="linear")(x)
            state_value = Dense(1, activation="linear")(x)
            network = keras.models.Model(inputs=inputs, outputs=[logits, state_value])

        return network

    def decision(self, state_space: np.ndarray, action_space: ActionSpace):
        """
        triggered by get play turn method of super class.
        This is the method were the magic should happen that chooses the right action
        :param state_space:
        :param action_space:
        :return:
        """
        #preprocess state space
        # normalizing state space between zero and one
        state_space = (state_space.astype('float32') - np.min(state_space)) / (np.max(state_space) - np.min(state_space))

        qvalues = self._get_qvalues([state_space])
        decision = self._sample_actions(qvalues, action_space)
        return decision

    def _sample_actions(self, qvalues: np.ndarray, action_space: ActionSpace):
        """
        pick actions given qvalues. Uses epsilon-greedy exploration strategy.
        :param qvalues: output values from network
        :param action_space: action_space with all possible actions
        :return: return stone_id and move_id
        """
        epsilon = self.epsilon
        batch_size, x = qvalues.shape
        dim = int(x ** 0.25)
        qvalues_reshaped = np.reshape(qvalues, (dim, dim, dim, dim))
        if random.random() < epsilon:
            keys = [key for key in action_space.keys()]
            ix = random.sample(range(len(keys)), 1)[0]
            stone_id = keys[ix]
            move_id = random.sample(range(len(action_space[stone_id])), 1)[0]
            move = action_space[stone_id][move_id]
            decision = np.concatenate([move["old_coord"], move["new_coord"]])
        else:
            possible_actions = qvalues_reshaped * action_space.space_array
            flattened_index = np.nanargmax(possible_actions)
            decision = np.array(np.unravel_index(flattened_index, self.action_shape))

        return decision

    def load_weigths_into_target_network(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        logging.debug("Transfer Weight!")
        assigns = []
        for w_agent, w_target in zip(self.weights, self.target_weights):
            assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
        self.sess.run(assigns)
        self.saver.save(self.sess, self._saver_path)

    def get_feedback(self, state, action, reward, next_state, finished):
        action_number = np.unravel_index(np.ravel_multi_index(action, self.action_shape), (4096,))[0]
        self.exp_buffer.add(state, action_number, reward, next_state, finished)
        if self.number_turns % self._intervall_actions_train == 0 and self.number_turns > 1:
            self.train_network()
        if self.number_turns % self._intervall_turns_load == 0 and self.number_turns > 1:
            self.load_weigths_into_target_network()

    def train_network(self):
        logging.debug("Train Network!")
        _, loss_t = self.sess.run([self._train_step, self._entropy], self._sample_batch(batch_size=self._batch_size))
        self.td_loss_history.append(loss_t)
        self.moving_average_loss.append(np.mean([self.td_loss_history[max([0, len(self.td_loss_history) - 100]):]]))
        ma = self.moving_average_loss[-1]
        relative_ma = self.moving_average_loss[-1] / self._batch_size
        logging.info("Loss: {},     relative Loss: {}".format(ma, relative_ma))

    def _get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.network(state_t)
        return qvalues

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
        self._actor_loss = - tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - 0.001 * \
                           tf.reduce_mean(self._entropy)
        target_state_values = self._rewards_ph + gamma * next_state_values
        self._critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values)) ** 2)

        self._train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(self._actor_loss + self._critic_loss )
        self.sess.run(tf.global_variables_initializer())

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = self.exp_buffer.sample(batch_size)
        obs_batch = obs_batch.reshape(obs_batch.shape[0], 1, obs_batch.shape[1] * obs_batch.shape[2])
        next_obs_batch = next_obs_batch.reshape(next_obs_batch.shape[0], 1, next_obs_batch.shape[1] * next_obs_batch.shape[2])
        return {
            self._obs_ph: obs_batch, self._actions_ph: act_batch, self._rewards_ph: reward_batch,
            self._next_obs_ph: next_obs_batch, self._is_done_ph: is_done_batch
        }