import tensorflow as tf
import numpy as np

from checkers.src.agents.QLearningAgent import QLearningAgent
from checkers.src.ReplayBuffer import ReplayBufferSarsa


class SARSAAgent(QLearningAgent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5,
                 intervall_turns_train: int = 500, intervall_turns_load: int = 10000):
        self._buffer_action = None
        self._buffer_state = None
        self._buffer_reward = None
        self._buffer_done = None

        super().__init__(state_shape=state_shape, action_shape=action_shape, name=name, side=side, epsilon=epsilon,
                         intervall_turns_train=intervall_turns_train, intervall_turns_load=intervall_turns_load)
        self._next_actions_ph = tf.placeholder(tf.int32, shape=[None])
        self.exp_buffer = ReplayBufferSarsa(100000)

    def get_feedback(self, state, action, reward, next_state, finished):
        if self._buffer_action is not None:
            action_number_buffer = np.unravel_index(np.ravel_multi_index(action, self.action_shape), (4096,))[0]
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

        # compute state values by taking max over next_qvalues_target for all actions
        next_state_values_target = tf.reduce_max(next_qvalues_target, axis=-1)

        # compute Q_reference(s,a) as per formula above.
        # TODO change value function here
        reference_qvalues = self._rewards_ph + gamma * next_state_values_target * is_not_done

        # Define loss function for sgd.
        td_loss = (current_action_qvalues - reference_qvalues) ** 2
        self._td_loss = tf.reduce_mean(td_loss)
        self.td_loss_history = []
        self._moving_average = []
        self._train_step = tf.train.AdamOptimizer(1e-3).minimize(self._td_loss, var_list=self.weights)
        self.sess.run(tf.global_variables_initializer())

    def train_network(self):
        _, loss_t = self.sess.run([self._train_step, self._td_loss], self._sample_batch(batch_size=512))
        self.td_loss_history.append(loss_t)
        self._moving_average.append(np.mean([self.td_loss_history[max([0,len(self.td_loss_history) - 100]):]]))
        print(self._moving_average[-1])

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, next_act_batch = self.exp_buffer.sample(batch_size)
        return {
            self._obs_ph: obs_batch, self._actions_ph: act_batch, self._rewards_ph: reward_batch,
            self._next_obs_ph: next_obs_batch, self._is_done_ph: is_done_batch, self._next_actions_ph: next_act_batch
        }