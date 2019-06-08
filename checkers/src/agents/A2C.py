import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LSTM, Input
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
                 save_path: str = "../data/modeldata/a2c/model.ckpt"):
        #TODO finalize
        # tensorflow related stuff
        self.name = name
        self._batch_size = 4096
        self._learning_rate = 0.3
        self._gamma = 0.99

        # calculate number actions from actionshape
        self.number_actions = np.product(action_shape)
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        self.target_network = self._configure_network(state_shape, "target_{}".format(name))
        self.network = self._configure_network(state_shape, self.name)

        self.epsilon = epsilon
        self.exp_buffer = ReplayBuffer(100000)

        self._save_path = save_path
        if os.path.isfile(self._save_path + ".index"):
            self.network.load_weights(self._save_path)

        # copy weight to target weights
        self.load_weigths_into_target_network()

        super().__init__(state_shape, action_shape, name, side)

    def _configure_network(self, state_shape: tuple, name: str):
        # define network
        inputs = Input(shape=(1, 64))
        x = LSTM(512, activation="relu", input_shape=(1, 64), return_sequences=True)(inputs)
        #x = LSTM(1024, activation="relu", return_sequences=True)(x)
        #x = LSTM(2048, activation="relu", return_sequences=True)(x)
        #x = Dense(4096, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)
        x = Flatten()(x)

        logits = Dense(self.number_actions, activation="linear")(x)
        state_value = Dense(1, activation="linear")(x)
        network = tf.keras.models.Sequential(inputs=inputs, outputs=[logits, state_value])
        return network

    def decision(self, state_space: np.ndarray, action_space: ActionSpace):
        """
        triggered by get play turn method of super class.
        This is the method were the magic should happen that chooses the right action
        :param state_space:
        :param action_space:
        :return:
        """
        # preprocess state space
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
        self.network.save_weights(self._save_path)
        self.target_network.load_weights(self._save_path)

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

    @tf.function
    def _train_network(self, obs, actions, next_obs, rewards, is_done):
        # placeholders that will be fed with exp_replay.sample(batch_size)

        qvalues, state_values = self._get_symbolic_qvalues(obs)
        next_qvalues, next_state_values = self.target_network(next_obs)
        next_state_values = next_state_values * (1 - is_done)
        probs = tf.nn.softmax(qvalues)
        logprobs = tf.nn.log_softmax(qvalues)

        logp_actions = tf.reduce_sum(logprobs * tf.one_hot(actions, self.number_actions), axis=-1)
        advantage = rewards + self._gamma * next_state_values - state_values
        entropy = -tf.reduce_sum(probs * logprobs, 1, name="entropy")
        actor_loss = - tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - 0.001 * \
                           tf.reduce_mean(entropy)
        target_state_values = rewards + self._gamma * next_state_values
        critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values)) ** 2)

        train_step = tf.optimizers.Adam(self._learning_rate).minimize(actor_loss + critic_loss)
        return train_step,

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = self.exp_buffer.sample(batch_size)
        obs_batch = obs_batch.reshape(obs_batch.shape[0], 1, obs_batch.shape[1] * obs_batch.shape[2])
        next_obs_batch = next_obs_batch.reshape(next_obs_batch.shape[0], 1, next_obs_batch.shape[1] *
                                                next_obs_batch.shape[2])
        return {"obs": obs_batch, "actions": act_batch, "rewards": reward_batch,
                "next_obs": next_obs_batch, "is_done": is_done_batch }