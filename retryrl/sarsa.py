"""Tabular SARSA (on-policy TD control) with epsilon-greedy exploration."""
from .qlearning import QLearningAgent


class SarsaAgent(QLearningAgent):
    """Same Q-table, action selection and save/load as Q-learning.
    Only the update differs: it bootstraps from the action actually taken next."""

    def update(self, s, a, r, s2, done, a2=None):
        # Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]
        target = r if done else r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])