import numpy as np

from .env import RetryEnv, ACTION_NAMES


def moving_average(x, w=500):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def evaluate_policy(policy_fn, n_episodes=3000, seed=12345, regime="normal"):
    """Run a policy (state -> action) and return averaged metrics."""
    env = RetryEnv(regime=regime)
    s, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rewards, served, lat, cost, acts, sources = [], [], [], [], np.zeros(4), {}
    for _ in range(n_episodes):
        done = False
        while not done:
            a = policy_fn(s, rng)
            s, r, done, _, info = env.step(a)
        ep = info["episode"]
        rewards.append(ep["reward"]); served.append(ep["served"])
        lat.append(ep["latency"]); cost.append(ep["cost"])
        acts += ep["actions"]
        sources[ep["source"]] = sources.get(ep["source"], 0) + 1
        s, _ = env.reset()
    acts = acts / acts.sum()
    return {
        "avg_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(served)),
        "avg_latency": float(np.mean(lat)),
        "avg_cost": float(np.mean(cost)),
        **{f"p_{n}": float(p) for n, p in zip(ACTION_NAMES, acts)},
    }
