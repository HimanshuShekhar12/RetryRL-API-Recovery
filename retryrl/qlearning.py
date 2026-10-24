"""Tabular Q-learning (off-policy TD control) with epsilon-greedy exploration."""
import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, seed=0):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.rng = np.random.default_rng(seed)

    def greedy(self, s):
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())       # random tie-break
        return int(self.rng.choice(best))

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.Q.shape[1]))
        return self.greedy(s)

    def update(self, s, a, r, s2, done):
        # Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
