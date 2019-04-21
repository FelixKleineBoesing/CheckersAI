from ..Agent import Agent
import random
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
sess = tf.InteractiveSession()

from keras.layers import Conv2D, Dense, Flatten
import keras


class QLearningAgent(Agent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5):
        """
        Agent which implements Q Learning
        :param state_shape: shape of state
        :param action_shape: shape of actions
        :param name: name of agent
        :param side: is he going to start at the top or at the bottom?
        :param epsilon: exploration factor
        """
        with tf.variable_scope(name, reuse=False):
            self.network = keras.models.Sequential()

            # let's create a network for approximate q-learning following guidelines above
            self.network.add(Dense(action_shape[0] * action_shape[1], activation="relu", input_shape=state_shape))
            self.network.add(Dense(64, activation="relu"))
            self.network.add(Dense(64, activation="relu"))
            self.network.add(Dense(64, activation="relu"))
            #self.network.add(Flatten())
            self.network.add(Dense(action_shape[0], activation="linear"))

            # prepare a graph for agent step
            self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
            self.qvalues_t = self._get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon
        sess.run(tf.global_variables_initializer())
        super().__init__(state_shape, action_shape, name, side)

    def decision(self, state_space: np.ndarray, action_space: dict):
        #preprocess state space
        state_space = (state_space.astype('float32') - np.min(state_space)) / (np.max(state_space) - np.min(state_space))
        qvalues = self._get_qvalues([state_space])
        stone_id, move_id = self._sample_actions(qvalues, action_space)
        # TODO return right stone and move id
        return {"stone_id": stone_id, "move_id": move_id}

    def _get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.qvalues_t, {self.state_t: state_t})

    def _sample_actions(self, qvalues, action_space):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, x, y = qvalues.shape
        if random.random() < epsilon:
            keys = [key for key in action_space.keys()]
            ix = random.sample(range(len(keys)), 1)[0]
            stone_id = keys[ix]
            move_id = random.sample(range(len(action_space[stone_id])), 1)[0]
        else:
            # TODO order decreasing by a value and choose one of those actions with the highest prob and random stone
            xx, yy = np.meshgrid(np.arange(qvalues.shape[2]), np.arange(qvalues.shape[1]))
            table = np.vstack((qvalues.ravel(), xx.ravel(), yy.ravel())).T
            table = table[table[:,0].argsort()]
            i = -1
            found = False
            while True:
                for stone in action_space.keys():
                    j = 0
                    for action in action_space[stone]:
                        coord = action["new_coord"]
                        if coord[0] == int(table[i, 1]) and coord[1] == int(table[i, 2]):
                            stone_id = stone
                            move_id = j
                            found = True
                            break
                        else:
                            j += 1
                    if found:
                        break
                if found:
                    break
                i -= 1

        return stone_id, move_id

    def _get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.network(state_t)
        return qvalues


class Learner:

    def __init__(self, network):
        """
        Learner, which takes the states and actions from Agent and learn with the same network
        :param network:
        """
        self.network = network


    def train_network(self, states: np.ndarray, actions: np.ndarray):
        """
        trains network with states and actions from Agent
        :param states:
        :param actions:
        :return:
        """
        pass

    def return_network_weights(self):
        """
        returns network weights for updating of Agent
        :return:
        """
        pass