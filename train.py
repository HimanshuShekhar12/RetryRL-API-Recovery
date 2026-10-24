"""Train the tabular Q-learning agent.

Usage:  python train.py --episodes 50000 --seed 0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from retryrl.env import RetryEnv, N_STATES, N_ACTIONS
from retryrl.qlearning import QLearningAgent
from retryrl.utils import moving_average


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = RetryEnv(regime="normal")
    agent = QLearningAgent(N_STATES, N_ACTIONS, alpha=args.alpha,
                           gamma=args.gamma, eps_end=args.eps_end, seed=args.seed)
    # exponential decay: epsilon reaches eps_end after 60% of training
    decay = args.eps_end ** (1.0 / (0.6 * args.episodes))

    s, _ = env.reset(seed=args.seed)
    log = []
    for ep in tqdm(range(args.episodes), desc="Q-learning"):
        eps = max(args.eps_end, decay ** ep)
        done = False
        while not done:
            a = agent.act(s, eps)
            s2, r, done, _, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
        e = info["episode"]
        log.append((ep, e["reward"], float(e["served"]), e["latency"], e["cost"], eps))
        s, _ = env.reset()

    df = pd.DataFrame(log, columns=["episode", "reward", "served", "latency", "cost", "epsilon"])
    df.to_csv(f"results/train_log_qlearning_seed{args.seed}.csv", index=False)
    agent.save(f"models/qlearning_seed{args.seed}.npy")

    ma = moving_average(df["reward"].values, 500)
    plt.figure(figsize=(6, 3.5))
    plt.plot(np.arange(len(ma)) + 500, ma)
    plt.xlabel("Episode"); plt.ylabel("Reward (500-ep moving avg)")
    plt.title("Q-learning training curve"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"results/learning_curve_qlearning_seed{args.seed}.png", dpi=150)
    print(f"Saved models/qlearning_seed{args.seed}.npy and learning curve. "
          f"Last-2000-episode avg reward: {df['reward'].tail(2000).mean():.2f}")


if __name__ == "__main__":
    main()
