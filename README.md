# RetryRL: Reinforcement Learning for Intelligent API Recovery

AI Course Mini Project (AI-A2)
Team: <Himanshu Shekhar B23MT1021>, <Amrutanshu Sahoo B23MT1009>

An RL agent that, after an API call fails, chooses between **RETRY, WAIT (backoff),
USE_CACHE, FALLBACK** to maximise reliability while minimising latency and cost.
The problem is modelled as an **MDP (S, A, P, R, gamma)** and solved with two
model-free temporal-difference methods: **tabular Q-learning** and **SARSA**.
Rule-based policies are used **only as comparison baselines**, not as the solution.

## MDP summary
| Item | Definition |
|---|---|
| S | 864 discrete states: last outcome (3) x failure-rate bucket (3) x latency bucket (3) x cache freshness (4) x failed attempts (4) x fallback tried (2) |
| A | 4 discrete actions: RETRY, WAIT, USE_CACHE, FALLBACK |
| P | Unknown to the agent. Hidden 3-state Markov chain (HEALTHY / DEGRADED / OUTAGE) drives failures, plus 429 rate-limit pressure, lognormal latency and cache ageing |
| R | +10 for a delivered response (scaled by source quality) - 0.5 x latency - 1.0 x cost - staleness penalty - 3 x [429] - 10 x [request abandoned] |
| gamma | 0.95 |

Q-learning: `Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]`
SARSA:      `Q(s,a) <- Q(s,a) + alpha [ r + gamma Q(s',a') - Q(s,a) ]`

## Installation
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick check (uses the trained agents already in models/)
```bash
python evaluate.py --episodes 2000
```

## Reproduce everything
```bash
# 1. train 5 seeds of each algorithm (about 2 minutes per run)
for s in 0 1 2 3 4; do python train.py --algo qlearning --seed $s; done
for s in 0 1 2 3 4; do python train.py --algo sarsa --seed $s; done

# 2. evaluate in the training regime and in an unseen, harsher regime
python evaluate.py --episodes 10000
python evaluate.py --regime harsh --episodes 10000

# 3. generate all figures and summary tables (needs step 1's training logs)
python make_figures.py
```
On Windows cmd, use `for %s in (0 1 2 3 4) do python train.py --algo qlearning --seed %s`.
Training logs (`results/train_log_*.csv`) are not committed because of their size;
run step 1 to regenerate them before running `make_figures.py`.
All randomness is seeded, so results reproduce exactly.

## Results (5 seeds, mean +- std, 10,000 evaluation episodes)
| Regime | Best baseline reward | Q-learning | SARSA |
|---|---|---|---|
| Normal | 6.85 (ThresholdFallback) | 6.99 +- 0.04 | 6.97 +- 0.04 |
| Harsh (unseen) | 4.62 (CacheFirst) | 5.84 +- 0.01 | 5.81 +- 0.01 |

## Structure
```
retryrl/env.py         Gymnasium environment and state encoder
retryrl/baselines.py   rule-based comparison policies (not RL)
retryrl/qlearning.py   tabular Q-learning agent
retryrl/sarsa.py       tabular SARSA agent
retryrl/utils.py       evaluation helpers
train.py               training script
evaluate.py            evaluation script
make_figures.py        figures and summary tables
models/                trained Q-tables (.npy)
results/               figures and metrics
```