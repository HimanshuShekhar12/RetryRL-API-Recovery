import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retryrl.baselines import BASELINES
from retryrl.env import (RetryEnv, decode_state, encode_state, ACTION_NAMES,
                         HEALTH_NAMES, USE_CACHE)
from retryrl.utils import moving_average

ALGOS = [("qlearning", "Q-learning", "tab:blue"), ("sarsa", "SARSA", "tab:orange")]
BASE_ORDER = ["AlwaysRetry", "ExpBackoff", "CacheFirst", "ThresholdFallback", "Random"]
os.makedirs("results", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"results/{name}.png", dpi=200)
    fig.savefig(f"results/{name}.pdf")
    plt.close(fig)
    print("saved", name)


def models_for(algo):
    return sorted(glob.glob(f"models/{algo}_seed*.npy"))


# ------------------------------------------------ summary tables (mean +- std)
def load_eval(regime):
    df = pd.read_csv(f"results/eval_{regime}.csv", index_col=0)
    rows = {b: df.loc[b] for b in BASE_ORDER if b in df.index}
    for algo, label, _ in ALGOS:
        prefix = {"Q-learning": "QLearning", "SARSA": "SARSA"}[label]
        sub = df[[i.startswith(prefix + "_s") for i in df.index]]
        if len(sub):
            rows[label] = sub
    return df, rows


def summary(regime):
    df, rows = load_eval(regime)
    out = []
    for name, v in rows.items():
        if isinstance(v, pd.DataFrame):
            m, s = v.mean(), v.std()
            out.append({"agent": f"{name} (n={len(v)})",
                        **{k: f"{m[k]:.2f} +- {s[k]:.2f}" for k in
                           ["avg_reward", "success_rate", "avg_latency", "avg_cost"]}})
        else:
            out.append({"agent": name, **{k: f"{v[k]:.2f}" for k in
                        ["avg_reward", "success_rate", "avg_latency", "avg_cost"]}})
    t = pd.DataFrame(out)
    t.to_csv(f"results/summary_{regime}.csv", index=False)
    print(f"\n=== {regime} regime ===\n{t.to_string(index=False)}")
    return rows


# ------------------------------------------------ Fig 1: learning curves
def fig_learning_curves(rows_normal):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    w = 2000
    for algo, label, col in ALGOS:
        curves = []
        for f in sorted(glob.glob(f"results/train_log_{algo}_seed*.csv")):
            curves.append(moving_average(pd.read_csv(f)["reward"].values, w))
        if not curves:
            continue
        C = np.array(curves)
        x = np.arange(C.shape[1]) + w
        mu, sd = C.mean(0), C.std(0)
        ax.plot(x[::500], mu[::500], color=col, label=f"{label} ({len(curves)} seeds)")
        ax.fill_between(x[::500], (mu - sd)[::500], (mu + sd)[::500], color=col, alpha=0.2)
    thr = rows_normal["ThresholdFallback"]["avg_reward"]
    ax.axhline(thr, color="gray", ls="--", label="ThresholdFallback (best baseline)")
    ax.set_xlabel("Training episode"); ax.set_ylabel(f"Reward ({w}-ep moving avg)")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig1_learning_curves")


# ------------------------------------------------ Fig 2: performance (normal)
def fig_performance(rows):
    metrics = [("avg_reward", "Avg reward (higher better)"),
               ("avg_latency", "Latency, s (lower better)"),
               ("avg_cost", "Cost (lower better)")]
    names = list(rows.keys())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (k, title) in zip(axes, metrics):
        vals, errs, cols = [], [], []
        for n in names:
            v = rows[n]
            if isinstance(v, pd.DataFrame):
                vals.append(v[k].mean()); errs.append(v[k].std())
                cols.append("tab:blue" if n == "Q-learning" else "tab:orange")
            else:
                vals.append(v[k]); errs.append(0); cols.append("lightgray")
        ax.bar(range(len(names)), vals, yerr=errs, color=cols, capsize=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8)
    save(fig, "fig2_performance_normal")


# ------------------------------------------------ Fig 3: robustness
def fig_robustness(rows_n, rows_h):
    names = list(rows_n.keys())

    def val(rows, n):
        v = rows[n]["avg_reward"]
        return (v.mean(), v.std()) if isinstance(rows[n], pd.DataFrame) else (v, 0)

    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for off, rows, lab, col in [(-w / 2, rows_n, "Normal (train regime)", "tab:green"),
                                (w / 2, rows_h, "Harsh (unseen)", "tab:red")]:
        mv = [val(rows, n) for n in names]
        ax.bar(x + off, [m for m, _ in mv], w, yerr=[s for _, s in mv],
               label=lab, color=col, alpha=0.8, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Avg reward"); ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=7)
    save(fig, "fig3_robustness")


# ------------------------------------------------ Fig 4: learned-policy heatmaps
def fig_policy_heatmap():
    path = models_for("qlearning")
    if not path:
        return
    Q = np.load(path[0])                          # seed 0
    cache_lbl = ["none", "fresh", "aging", "stale"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for ax, (fail, title) in zip(axes, [(0, "Low recent failure rate"),
                                        (2, "High recent failure rate")]):
        grid = np.zeros((4, 4), dtype=int)       # rows: retry bucket, cols: cache
        for retry in range(4):
            for cache in range(4):
                s = encode_state(0, fail, 1, cache, retry, 0)   # last=timeout, mid latency
                grid[retry, cache] = int(np.argmax(Q[s]))
        ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(colors), vmin=0, vmax=3)
        for r in range(4):
            for c in range(4):
                ax.text(c, r, ACTION_NAMES[grid[r, c]].replace("USE_", "").replace("FALLBACK", "FALL-\nBACK"),
                        ha="center", va="center", color="white", fontsize=6.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(cache_lbl)
        ax.set_yticks(range(4)); ax.set_yticklabels(["1", "2", "3", "4+"])
        ax.set_xlabel("Cache freshness"); ax.set_ylabel("Failed attempts so far")
        ax.set_title(title, fontsize=8); ax.grid(False)
    save(fig, "fig4_policy_heatmap")


# ------------------------------------------------ Fig 5: actions by TRUE health
def actions_by_health(policy_fn, episodes=3000, seed=777):
    env = RetryEnv(regime="normal")
    s, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    counts = np.zeros((3, 4))
    for _ in range(episodes):
        done = False
        while not done:
            h = env.health                         # hidden to the agent, used for analysis only
            a = policy_fn(s, rng)
            counts[h, a] += 1
            s, _, done, _, _ = env.step(a)
        s, _ = env.reset()
    return counts / np.maximum(counts.sum(1, keepdims=True), 1)


def fig_actions_by_health():
    qs = [np.load(p) for p in models_for("qlearning")]
    if not qs:
        return
    fr_q = np.mean([actions_by_health(lambda s, r, Q=Q: int(np.argmax(Q[s]))) for Q in qs], axis=0)
    fr_t = actions_by_health(BASELINES["ThresholdFallback"])
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9), sharey=True)
    for ax, fr, title in [(axes[0], fr_t, "ThresholdFallback (baseline)"),
                          (axes[1], fr_q, "Q-learning (learned)")]:
        bottom = np.zeros(3)
        for a in range(4):
            ax.bar(HEALTH_NAMES, fr[:, a], bottom=bottom, color=colors[a], label=ACTION_NAMES[a])
            bottom += fr[:, a]
        ax.set_title(title, fontsize=8); ax.set_xlabel("True service health (hidden)")
    axes[0].set_ylabel("Fraction of actions"); axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4)
    save(fig, "fig5_actions_by_health")
    print("\nQ-learning action mix by true health (rows H/D/O; RETRY/WAIT/CACHE/FALLBACK):")
    print(np.round(fr_q, 2))


# ------------------------------------------------ sanity: invalid greedy actions
def sanity_check():
    print("\nSanity check: visited states with EMPTY cache whose greedy action is USE_CACHE")
    for algo, label, _ in ALGOS:
        for p in models_for(algo):
            Q = np.load(p)
            bad = 0
            for s in np.flatnonzero(np.abs(Q).sum(1) > 0):
                if decode_state(s)[3] == 0 and int(np.argmax(Q[s])) == USE_CACHE:
                    bad += 1
            print(f"  {os.path.basename(p):24s} {bad}")


if __name__ == "__main__":
    rows_n = summary("normal")
    rows_h = summary("harsh")
    fig_learning_curves(rows_n)
    fig_performance(rows_n)
    fig_robustness(rows_n, rows_h)
    fig_policy_heatmap()
    fig_actions_by_health()
    sanity_check()