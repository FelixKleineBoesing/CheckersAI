import numpy as np
import tensorflow as tf
import random
from keras.layers import Dense, Flatten
import keras
import os

from checkers.src.agents.Agent import Agent
from checkers.src.Helpers import ActionSpace
from checkers.src.ReplayBuffer import ReplayBuffer


class QLearningAgent(Agent):

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
        tf.reset_default_graph()
        self.name = name
        self.sess = tf.InteractiveSession()

        # calculate number actions from actionshape
        self.number_actions = np.product(action_shape)
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        self.target_network = self._configure_network(state_shape)
        self.network = self._configure_network(state_shape)

        # prepare a graph for agent step
        self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
        self.qvalues_t = self._get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon
        self.target_weights = self.weights
        self.exp_buffer = ReplayBuffer(100000)

        # init placeholder
        self._obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_shape)
        self._actions_ph = tf.placeholder(tf.int32, shape=[None])
        self._rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self._next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_shape)
        self._is_done_ph = tf.placeholder(tf.float32, shape=[None])

        self.saver = tf.train.Saver()
        self._configure_target_model()
        if os.path.isfile("../data/modeldata/model.ckpt.index"):
            self.saver.restore(self.sess, "../data/modeldata/model.ckpt")

        # copy weight to target weights

        self.load_weigths_into_target_network()

        super().__init__(state_shape, action_shape, name, side)

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

    def _get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        return self.sess.run(self.qvalues_t, {self.state_t: state_t})

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
        if self._number_turns % self._intervall_actions_train == 0 and self._number_turns > 1:
            self.train_network()
        if self._number_turns % self._intervall_turns_load == 0 and self._number_turns > 1:
            self.load_weigths_into_target_network()

    def _get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.network(state_t)
        return qvalues

    def _configure_target_model(self):
        # placeholders that will be fed with exp_replay.sample(batch_size)
        is_not_done = 1 - self._is_done_ph
        gamma = 0.99
        current_qvalues = self._get_symbolic_qvalues(self._obs_ph)
        current_action_qvalues = tf.reduce_sum(tf.one_hot(self._actions_ph, self.number_actions) * current_qvalues, axis=1)

        # compute q-values for NEXT states with target network
        next_qvalues_target = self.target_network(self._next_obs_ph)

        # compute state values by taking max over next_qvalues_target for all actions
        next_state_values_target = tf.reduce_max(next_qvalues_target, axis=-1)

        # compute Q_reference(s,a) as per formula above.
        reference_qvalues = self._rewards_ph + gamma * next_state_values_target * is_not_done

        # Define loss function for sgd.
        td_loss = (current_action_qvalues - reference_qvalues) ** 2
        self._td_loss = tf.reduce_mean(td_loss)
        self.td_loss_history = []
        self._moving_average = []
        self._train_step = tf.train.AdamOptimizer(1e-3).minimize(self._td_loss, var_list=self.weights)
        self.sess.run(tf.global_variables_initializer())

    def load_weigths_into_target_network(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        assigns = []
        for w_agent, w_target in zip(self.weights, self.target_weights):
            assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
        self.sess.run(assigns)
        self.saver.save(self.sess, "../data/modeldata/model.ckpt")

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = self.exp_buffer.sample(batch_size)
        return {
            self._obs_ph: obs_batch, self._actions_ph: act_batch, self._rewards_ph: reward_batch,
            self._next_obs_ph: next_obs_batch, self._is_done_ph: is_done_batch
        }

    def train_network(self):
        _, loss_t = self.sess.run([self._train_step, self._td_loss], self._sample_batch(batch_size=512))
        self.td_loss_history.append(loss_t)
        self._moving_average.append(np.mean([self.td_loss_history[max([0,len(self.td_loss_history) - 100]):]]))

    def _configure_network(self, state_shape: tuple):
        # define network
        with tf.variable_scope(self.name, reuse=False):
            network = keras.models.Sequential()
            # TODO try agent performance with rnn layers (perhaps LSTM since we have sequential dependencies in the data)
            network.add(Dense(512, activation="relu", input_shape=state_shape))
            #network.add(Dense(1024, activation="relu"))
            #network.add(Dense(2048, activation="relu"))
            network.add(Dense(4096, activation="relu"))
            network.add(Flatten())
            network.add(Dense(self.number_actions, activation="linear"))
        return network
