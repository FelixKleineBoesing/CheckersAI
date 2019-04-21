from ..Agent import Agent
import random
import numpy as np
import tensorflow as tf
import random
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


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
