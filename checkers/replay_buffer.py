import numpy as np
import random


class ReplayBuffer:

    def __init__(self, size: int = 10000):
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


class ReplayBufferSarsa(ReplayBuffer):

    def add(self, obs_t, action, reward, obs_tp1, done, next_actions=None):
        data = (obs_t, action, reward, obs_tp1, done, next_actions)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, next_actions = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, next_states = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            next_actions.append(next_states)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), \
               np.array(next_actions)


class EpisodeBuffer:

    """
    This Buffer implementation stores and samples whole episodes instead of random samples from multiple episodes
    which comes in handy when you use recurrent neural networks which rely on detecting time dependend connections
    """
    def __init__(self, size: int = 1000):
        """
        ATTENTION! Size is now related to the amount of episodes that should be stored - not (state, action)-tuples
        :param size:
        """
        self._episode_number = 0
        self._episode_length = {}
        self._buffer_storage = []
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, next_actions=None):
        data = (obs_t, action, reward, obs_tp1, done, next_actions)
        self._buffer_storage.append(data)
        if done:
            if self._next_idx >= len(self._storage):
                self._storage.append(self._buffer_storage)
            else:
                self._storage[self._next_idx] = self._buffer_storage
            self._next_idx = (self._next_idx + 1) % self._maxsize
            self._episode_number += 1
            self._buffer_storage = []

    def _encode_sample(self, batch_size):
        obses_t, actions, rewards, obses_tp1, dones, next_actions = [], [], [], [], [], []
        enough = False
        while not enough:
            i = random.randint(0, len(self._storage) - 1)
            data = self._storage[i]
            for tmp in data:
                obs_t, action, reward, obs_tp1, done, next_states = tmp
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
                next_actions.append(next_states)
            enough = True if len(dones) >= batch_size else False

        obses_t = obses_t[0:batch_size]
        actions = actions[0:batch_size]
        rewards = rewards[0:batch_size]
        obses_tp1 = obses_tp1[0:batch_size]
        dones = dones[0:batch_size]
        next_actions = next_actions[0:batch_size]

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), \
               np.array(next_actions)

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
        return self._encode_sample(batch_size)