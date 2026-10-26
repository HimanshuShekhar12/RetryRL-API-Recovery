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
from retryrl.sarsa import SarsaAgent
from retryrl.utils import moving_average


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["qlearning", "sarsa"], default="qlearning")
    p.add_argument("--episodes", type=int, default=200000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = RetryEnv(regime="normal")
    Agent = QLearningAgent if args.algo == "qlearning" else SarsaAgent
    agent = Agent(N_STATES, N_ACTIONS, alpha=args.alpha, gamma=args.gamma,
                  eps_end=args.eps_end, seed=args.seed)
    # exponential decay: epsilon reaches eps_end after 60% of training
    decay = args.eps_end ** (1.0 / (0.6 * args.episodes))

    s, _ = env.reset(seed=args.seed)
    log = []
    for ep in tqdm(range(args.episodes), desc=args.algo):
        eps = max(args.eps_end, decay ** ep)
        done = False
        a = agent.act(s, eps)
        while not done:
            s2, r, done, _, info = env.step(a)
            if args.algo == "sarsa":
                a2 = agent.act(s2, eps)          # next action chosen BEFORE the update
                agent.update(s, a, r, s2, done, a2)
                s, a = s2, a2
            else:
                agent.update(s, a, r, s2, done)
                s = s2
                a = agent.act(s, eps)
        e = info["episode"]
        log.append((ep, e["reward"], float(e["served"]), e["latency"], e["cost"], eps))
        s, _ = env.reset()

    tag = f"{args.algo}_seed{args.seed}"
    df = pd.DataFrame(log, columns=["episode", "reward", "served", "latency", "cost", "epsilon"])
    df.to_csv(f"results/train_log_{tag}.csv", index=False)
    agent.save(f"models/{tag}.npy")

    ma = moving_average(df["reward"].values, 500)
    plt.figure(figsize=(6, 3.5))
    plt.plot(np.arange(len(ma)) + 500, ma)
    plt.xlabel("Episode"); plt.ylabel("Reward (500-ep moving avg)")
    plt.title(f"{args.algo} training curve"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"results/learning_curve_{tag}.png", dpi=150)
    print(f"Saved models/{tag}.npy and learning curve. "
          f"Last-2000-episode avg reward: {df['reward'].tail(2000).mean():.2f}")


if __name__ == "__main__":
    main()