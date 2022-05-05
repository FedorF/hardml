import random

import numpy as np


class Strategy:

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.n_iters = 0
        self.arms_states = np.zeros(n_arms)
        self.arms_actions = np.zeros(n_arms)

    def flush(self):
        self.n_iters = 0
        self.arms_states = np.zeros(self.n_arms)
        self.arms_actions = np.zeros(self.n_arms)

    def update_reward(self, arm: int, reward: int):
        self.n_iters += 1
        self.arms_states[arm] += reward
        self.arms_actions[arm] += 1

    def choose_arm(self):
        raise NotImplementedError


class EpsGreedy(Strategy):

    def __init__(self, n_arms: int, eps: float = 0.1):
        super().__init__(n_arms)
        self.eps = eps

    def choose_arm(self):

        if random.random() < self.eps:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.arms_states / self.arms_actions)


class UCB1(Strategy):

    def choose_arm(self):
        if self.n_iters < self.n_arms:
            return self.n_iters
        else:
            return np.argmax(self.ucb())

    def ucb(self):
        ucb = self.arms_states / self.arms_actions
        ucb += np.sqrt(2 * np.log(self.n_iters) / self.arms_actions)
        return ucb


class Thompson(Strategy):

    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.a = np.ones(n_arms)
        self.b = np.ones(n_arms)
        self.theta = np.zeros(n_arms)

    def update_reward(self, arm: int, reward: int):
        self.a[arm] += reward
        self.b[arm] += 1 - reward

        self.n_iters += 1
        self.arms_states[arm] += reward
        self.arms_actions[arm] += 1

    def choose_arm(self):
        self.theta = np.random.beta(self.a, self.b)

        return np.argmax(self.theta)
