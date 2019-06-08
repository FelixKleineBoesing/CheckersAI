import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
import random
import os
import logging

from checkers.src.agents.Agent import Agent
from checkers.src.Helpers import ActionSpace
from checkers.src.ReplayBuffer import ReplayBuffer
from checkers.src.cache.RedisWrapper import RedisChannel, RedisCache
from checkers.src.Helpers import Config


class QLearningAgent(Agent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5,
                 intervall_turns_train: int = 500, intervall_turns_load: int = 10000,
                 save_path: str = "../data/modeldata/q/model.ckpt", caching: bool = False,
                 config: Config = None, cache: RedisCache = None, channel: RedisChannel = None):
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
        self._batch_size = 4096
        self._learning_rate = 0.3
        self._gamma = 0.99

        # calculate number actions from actionshape
        self.number_actions = np.product(action_shape)
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        self.network = self._configure_network(state_shape, self.name)
        self.target_network = self._configure_network(state_shape, "target_{}".format(name))

        self.epsilon = epsilon
        self.exp_buffer = ReplayBuffer(100000)

        self._save_path = save_path
        if os.path.isfile(self._save_path + ".index"):
            self.network.load_weights(self._save_path)

        # copy weight to target weights
        self.load_weigths_into_target_network()

        super().__init__(state_shape, action_shape, name, side, config, caching, cache, channel)

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

    def get_feedback(self, state, action, reward, next_state, finished):
        action_number = np.unravel_index(np.ravel_multi_index(action, self.action_shape), (4096,))[0]
        self.exp_buffer.add(state, action_number, reward, next_state, finished)
        if self.number_turns % self._intervall_actions_train == 0 and self.number_turns > 1:
            self.train_network()
        if self.number_turns % self._intervall_turns_load == 0 and self.number_turns > 1:
            self.load_weigths_into_target_network()

    def _get_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.network(state_t)
        return qvalues

    def load_weigths_into_target_network(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        logging.debug("Transfer Weight!")
        self.network.save_weights(self._save_path)
        self.target_network.load_weights(self._save_path)

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = self.exp_buffer.sample(batch_size)
        return {"obs": obs_batch, "actions": act_batch, "rewards": reward_batch,
                "next_obs": next_obs_batch, "is_done": is_done_batch}

    def train_network(self):
        logging.debug("Train Network!")
        _, loss_t = self._train_network(self._sample_batch(batch_size=self._batch_size))
        self.td_loss_history.append(loss_t)
        self.moving_average_loss.append(np.mean([self.td_loss_history[max([0, len(self.td_loss_history) - 100]):]]))
        ma = self.moving_average_loss[-1]
        relative_ma = self.moving_average_loss[-1] / self._batch_size
        logging.info("Loss: {},     relative Loss: {}".format(ma, relative_ma))
        if self.redis_cache is not None:
            self.publish_data()

    def _configure_network(self, state_shape: tuple, name: str):

        network = tf.keras.models.Sequential([
            Dense(512, activation="relu", input_shape=state_shape),
            # Dense(1024, activation="relu"),
            # Dense(2048, activation="relu"),
            # Dense(4096, activation="relu"),
            Dense(2048, activation="relu"),
            Flatten(),
            Dense(self.number_actions, activation="linear")])
        return network

    @tf.function
    def _train_network(self, obs, actions, next_obs, rewards, is_done):
        current_qvalues = self._get_qvalues(obs)
        current_action_qvalues = tf.reduce_sum(tf.one_hot(actions, self.number_actions) * current_qvalues, axis=1)

        # compute q-values for NEXT states with target network
        next_qvalues_target = self.target_network(next_obs)
        next_state_values_target = tf.reduce_max(next_qvalues_target, axis=-1)
        reference_qvalues = rewards + self._gamma * next_state_values_target * (1 - is_done)

        # Define loss function for sgd.
        td_loss = tf.reduce_mean(current_action_qvalues - reference_qvalues) ** 2
        # TODO deliver var_list in ada optimizer
        train_step = tf.optimizers.Adam(self._learning_rate).minimize(td_loss,
                                                                      var_list=self.network.trainable_variables)
        return train_step, td_loss
