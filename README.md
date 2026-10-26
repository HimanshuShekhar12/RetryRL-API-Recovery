# RetryRL: Reinforcement Learning for Intelligent API Recovery

AI Course Mini Project (AI-A2)

**Team:** Himanshu Shekhar (B23MT1021), Amrutanshu Sahoo (B23MT1009)

---

## Overview

When an API call fails, what should a client do next: retry immediately, back off,
serve stale data from a cache, or switch to a fallback service? Fixed rules handle
this poorly because the right answer depends on the (hidden) health of the service.

RetryRL frames this as a sequential decision problem and trains an agent that, after
each failure, chooses between four recovery actions to **maximise reliability while
minimising latency and cost**. The problem is modelled as an **MDP (S, A, P, R, gamma)**
and solved with two model-free temporal-difference methods:

- **Tabular Q-learning** (off-policy)
- **SARSA** (on-policy)

Rule-based policies are used **only as comparison baselines**, not as part of the solution.

## Key results

Mean +- std over 5 training seeds, 10,000 evaluation episodes per run.

| Regime | Best baseline reward | Q-learning | SARSA |
|---|---|---|---|
| Normal (training regime) | 6.85 (ThresholdFallback) | **6.99 +- 0.04** | 6.97 +- 0.04 |
| Harsh (unseen outages) | 4.62 (CacheFirst) | **5.84 +- 0.01** | 5.81 +- 0.01 |

Both agents beat every rule-based baseline in the training regime, and the gap widens
in the unseen harsh regime, which suggests the learned policies generalise better
than hand-written rules.

## Problem formulation (MDP)

| Item | Definition |
|---|---|
| **S** | 864 discrete states: last outcome (3) x failure-rate bucket (3) x latency bucket (3) x cache freshness (4) x failed attempts (4) x fallback tried (2) |
| **A** | 4 discrete actions: `RETRY`, `WAIT` (backoff), `USE_CACHE`, `FALLBACK` |
| **P** | Unknown to the agent. A hidden 3-state Markov chain (HEALTHY / DEGRADED / OUTAGE) drives failures, combined with 429 rate-limit pressure, lognormal latency and cache ageing |
| **R** | +10 for a delivered response (scaled by source quality) - 0.5 x latency - 1.0 x cost - staleness penalty - 3 x [429] - 10 x [request abandoned] |
| **gamma** | 0.95 |

The agent never observes the true service health. It only sees the discretised
signals in the state, so it must infer the service condition from recent history.

### Update rules

```
Q-learning:  Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
SARSA:       Q(s,a) <- Q(s,a) + alpha [ r + gamma * Q(s',a')        - Q(s,a) ]
```

### Training setup

| Hyperparameter | Default |
|---|---|
| Episodes | 200,000 |
| Learning rate (alpha) | 0.05 |
| Discount (gamma) | 0.95 |
| Exploration | epsilon-greedy, exponential decay to 0.05, reached after 60% of training |

## Baselines

Used for comparison only:

| Baseline | Idea |
|---|---|
| `AlwaysRetry` | Retry on every failure |
| `ExpBackoff` | Retry with exponential backoff |
| `CacheFirst` | Prefer cached data |
| `ThresholdFallback` | Retry up to a threshold, then fall back |
| `Random` | Uniformly random action |

## Evaluation metrics

Each agent is scored on `avg_reward`, `success_rate`, `avg_latency` and `avg_cost`.
Evaluation runs in two regimes:

- **normal**: the regime used for training
- **harsh**: an unseen regime with harsher outages, used to test robustness

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Quick check

Evaluates the trained agents already stored in `models/`:

```bash
python evaluate.py --episodes 2000
```

### Reproduce everything

```bash
# 1. Train 5 seeds of each algorithm (about 2 minutes per run)
for s in 0 1 2 3 4; do python train.py --algo qlearning --seed $s; done
for s in 0 1 2 3 4; do python train.py --algo sarsa --seed $s; done

# 2. Evaluate in the training regime and in the unseen, harsher regime
python evaluate.py --episodes 10000
python evaluate.py --regime harsh --episodes 10000

# 3. Generate all figures and summary tables (needs the training logs from step 1)
python make_figures.py
```

On Windows cmd:

```bat
for %s in (0 1 2 3 4) do python train.py --algo qlearning --seed %s
```

Notes:

- Training logs (`results/train_log_*.csv`) are not committed because of their size.
  Run step 1 to regenerate them before running `make_figures.py`.
- All randomness is seeded, so results reproduce exactly.

### Command-line options

| Script | Option | Default | Description |
|---|---|---|---|
| `train.py` | `--algo` | `qlearning` | `qlearning` or `sarsa` |
| `train.py` | `--episodes` | 200000 | Training episodes |
| `train.py` | `--alpha` | 0.05 | Learning rate |
| `train.py` | `--gamma` | 0.95 | Discount factor |
| `train.py` | `--eps_end` | 0.05 | Final exploration rate |
| `train.py` | `--seed` | 0 | Random seed |
| `evaluate.py` | `--regime` | `normal` | `normal` or `harsh` |
| `evaluate.py` | `--episodes` | 3000 | Evaluation episodes |

## Outputs

`make_figures.py` writes PNG and PDF files to `results/`:

| Figure | Content |
|---|---|
| `fig1_learning_curves` | Training reward for both algorithms vs. the best baseline |
| `fig2_performance_normal` | Reward, latency and cost per agent in the normal regime |
| `fig3_robustness` | Normal vs. harsh regime reward |
| `fig4_policy_heatmap` | Learned Q-learning policy by cache freshness and failed attempts |
| `fig5_actions_by_health` | Action mix under each true service health state, learned vs. ThresholdFallback |

It also saves `summary_normal.csv` and `summary_harsh.csv`, and prints a sanity check
counting visited states with an empty cache whose greedy action is `USE_CACHE`.

## Project structure

```
retryrl/
  env.py            Gymnasium environment and state encoder
  baselines.py      Rule-based comparison policies (not RL)
  qlearning.py      Tabular Q-learning agent
  sarsa.py          Tabular SARSA agent
  utils.py          Evaluation helpers
train.py            Training script
evaluate.py         Evaluation script
make_figures.py     Figures and summary tables
models/             Trained Q-tables (.npy)
results/            Figures and metrics
```