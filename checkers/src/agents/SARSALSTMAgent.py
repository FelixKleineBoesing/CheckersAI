import tensorflow as tf
from keras.layers import Dense, Flatten, LSTM
import keras

from checkers.src.ReplayBuffer import EpisodeBuffer
from checkers.src.agents.SARSAAgent import SARSAAgent


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

        super().__init__(state_shape, action_shape, name, side, epsilon, intervall_turns_train, intervall_turns_load,
                         saver_path)
        self.exp_buffer = EpisodeBuffer(5000)

    def _configure_network(self, state_shape: tuple, name: str):
        # define network
        with tf.variable_scope(name, reuse=False):
            network = keras.models.Sequential()
            network.add(LSTM(512, activation="relu", batch_input_shape=(2048, 8, 8, 8), return_sequences=True))
            network.add(LSTM(1024, activation="relu", return_sequences=True))
            network.add(LSTM(512, activation="relu", return_sequences=True))
            network.add(Flatten())
            network.add(Dense(self.number_actions, activation="linear"))
        return network

    def _get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.network(state_t)
        return qvalues

    def _get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        return self.sess.run(self.qvalues_t, {self.state_t: state_t})

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, next_act_batch = self.exp_buffer.sample(batch_size)
        return {
            self._obs_ph: obs_batch, self._actions_ph: act_batch, self._rewards_ph: reward_batch,
            self._next_obs_ph: next_obs_batch, self._is_done_ph: is_done_batch, self._next_actions_ph: next_act_batch
        }