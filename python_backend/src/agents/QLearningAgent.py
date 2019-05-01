import numpy as np
import tensorflow as tf
import random
import copy
from keras.layers import Dense
import keras

from python_backend.src.agents.Agent import Agent
from python_backend.src.Helpers import ActionSpace


class QLearningAgent(Agent):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, side: str = "up", epsilon: float = 0.5,
                 intervall_turns_train: int = 20, intervall_turns_load = 500):
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
        self.sess = tf.InteractiveSession()

        # calculate number actions from actionshape
        self.number_actions = np.product(action_shape)
        self._intervall_actions_train = intervall_turns_train
        self._intervall_turns_load = intervall_turns_load

        # define network
        with tf.variable_scope(name, reuse=False):
            self.network = keras.models.Sequential()

            # let's create a network for approximate q-learning following guidelines above
            # input shape: X x Y (board size)
            self.network.add(Dense(512, activation="relu", input_shape=state_shape))
            self.network.add(Dense(1024, activation="relu"))
            self.network.add(Dense(2048, activation="relu"))
            self.network.add(Dense(4096, activation="relu"))
            #self.network.add(Flatten())
            # outpu shape: X_From x Y_From x X_To x Y_To (board size x action shape)
            self.network.add(Dense(self.number_actions, activation="linear"))

            # prepare a graph for agent step
            self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
            self.qvalues_t = self._get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon
        self.target_network = self.network
        self.target_weights = self.weights
        self.exp_buffer = ReplayBuffer(200)

        # init placeholder
        self._obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_shape)
        self._actions_ph = tf.placeholder(tf.int32, shape=[None])
        self._rewards_ph = tf.placeholder(tf.float32, shape=[None])
        self._next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_shape)
        self._is_done_ph = tf.placeholder(tf.float32, shape=[None])

        self._configure_target_model()
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
        stone_id, move_id = self._sample_actions(qvalues, action_space)
        # TODO return right stone and move id
        return {"stone_id": stone_id, "move_id": move_id}

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

    def get_feedback(self, state, action, reward, next_state, finished):
        self.exp_buffer.add(state, action, reward, next_state, finished)
        if self._number_turns % self._intervall_actions_train == 0:
            self.train_network()
        if self._number_turns % self._intervall_turns_load == 0:
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
        # TODO perhaps move target_network in own class
        next_qvalues_target = self.target_network(self._next_obs_ph)

        # compute state values by taking max over next_qvalues_target for all actions
        next_state_values_target = tf.reduce_max(next_qvalues_target, axis=-1)

        # compute Q_reference(s,a) as per formula above.
        reference_qvalues = self._rewards_ph + gamma * next_state_values_target * is_not_done

        # Define loss function for sgd.
        td_loss = (current_action_qvalues - reference_qvalues) ** 2
        self._td_loss = tf.reduce_mean(td_loss)

        self._train_step = tf.train.AdamOptimizer(1e-3).minimize(td_loss, var_list=self.weights)
        self.sess.run(tf.global_variables_initializer())


    def load_weigths_into_target_network(self):
        """ assign target_network.weights variables to their respective agent.weights values. """
        assigns = []
        for w_agent, w_target in zip(self.weights, self.target_weights):
            assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
        self.sess.run(assigns)

    def _sample_batch(self, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = self.exp_buffer.sample(batch_size)
        return {
            self._obs_ph: obs_batch, self._actions_ph: act_batch, self._rewards_ph: reward_batch,
            self._next_obs_ph: next_obs_batch, self._is_done_ph: is_done_batch
        }

    def train_network(self):
        _, loss_t = self.sess.run([self._train_step, self._td_loss], self._sample_batch( batch_size=64))



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
