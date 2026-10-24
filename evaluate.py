"""Evaluate baselines + trained Q-learning agents.

Usage:  python evaluate.py                 # normal regime
        python evaluate.py --regime harsh  # unseen harsher outages
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from retryrl.baselines import BASELINES
from retryrl.utils import evaluate_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="normal", choices=["normal", "harsh"])
    p.add_argument("--episodes", type=int, default=3000)
    args = p.parse_args()
    os.makedirs("results", exist_ok=True)

    rows = {}
    for name, fn in BASELINES.items():
        rows[name] = evaluate_policy(fn, args.episodes, regime=args.regime)

    for path in sorted(glob.glob("models/qlearning_seed*.npy")):
        Q = np.load(path)
        seed = os.path.basename(path).split("seed")[-1].split(".")[0]
        rows[f"QLearning_s{seed}"] = evaluate_policy(
            lambda s, rng, Q=Q: int(np.argmax(Q[s])), args.episodes, regime=args.regime)

    df = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print(df.round(3).to_string())
    df.to_csv(f"results/eval_{args.regime}.csv")


if __name__ == "__main__":
    main()
