import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time

import random
from collections import namedtuple, deque
import copy


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size , batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done","delta"])

    def add(self, state, action, reward, next_state, done,delta):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done,delta)
        self.memory.append(e)

    def sample(self, batch_size=64,prioritised_replay = False):
        """Randomly sample a batch of experiences from memory."""
        if not prioritised_replay: 
            return random.sample(self.memory, k=self.batch_size)
        else:
            e = 1e-4
            a = 1
            # prioritised experience replay
            deltas = np.array([e.delta for e in self.memory if e is not None])
            weight_delta = np.power(deltas + e,a)
            probs = weight_delta/np.sum(weight_delta)      
            idx =  np.random.choice(np.arange(len(self.memory)),size=self.batch_size,replace=False,p=probs)
            return [self.memory[ii] for ii in idx]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state