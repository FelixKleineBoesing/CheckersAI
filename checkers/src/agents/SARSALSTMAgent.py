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
            network.add(LSTM(512, activation="relu", input_shape=state_shape))
            network.add(LSTM(4096, activation="relu"))
            network.add(Flatten())
            network.add(Dense(self.number_actions, activation="linear"))
        return network
