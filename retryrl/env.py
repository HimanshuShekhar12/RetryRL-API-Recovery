"""
RetryRL environment: API failure recovery as an MDP.

One episode = one API request that has ALREADY failed once. The agent then
chooses, step by step, how to recover:

    0 RETRY      immediate re-call of the primary service
    1 WAIT       back off (time passes, service may recover), then re-call
    2 USE_CACHE  return the cached response (terminal; quality depends on age)
    3 FALLBACK   call the secondary (paid, slower, slightly lower quality) service

Hidden dynamics P (never shown to the agent):
    * Primary-service health follows a 3-state Markov chain
      HEALTHY / DEGRADED / OUTAGE  -> failures are temporally correlated.
    * Failure types: timeout, 5xx, 429 (rate limit). Hammering the API with
      back-to-back retries triggers 429s; WAIT resets that pressure.
    * Latency is lognormal on success, fixed per failure type.
    * The client keeps running statistics (failure-rate EMA, latency EMA) from
      ALL its traffic, so these persist across episodes and carry information
      about the hidden health.

Observation (discrete, |S| = 864):
    last_outcome (3) x failure_rate (3) x latency (3) x cache (4) x
    retries (4) x fallback_tried (2)
"""
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------- constants
RETRY, WAIT, USE_CACHE, FALLBACK = 0, 1, 2, 3
ACTION_NAMES = ["RETRY", "WAIT", "USE_CACHE", "FALLBACK"]

HEALTHY, DEGRADED, OUTAGE = 0, 1, 2
HEALTH_NAMES = ["HEALTHY", "DEGRADED", "OUTAGE"]

TIMEOUT, ERR5XX, RATE429, SUCCESS = 0, 1, 2, 3
OUTCOME_NAMES = ["timeout", "5xx", "429", "success"]

# State-feature sizes (order matters for encode/decode)
N_LAST, N_FAIL, N_LAT, N_CACHE, N_RETRY, N_FB = 3, 3, 3, 4, 4, 2
N_STATES = N_LAST * N_FAIL * N_LAT * N_CACHE * N_RETRY * N_FB  # 864
N_ACTIONS = 4

# ------------------------------------------------------- simulator regimes
# Row = current health, column = next health (one tick).
REGIMES = {
    "normal": np.array([[0.90, 0.07, 0.03],
                        [0.12, 0.78, 0.10],
                        [0.04, 0.08, 0.88]]),
    # harsher, never used in training: longer / more frequent outages
    "harsh": np.array([[0.80, 0.12, 0.08],
                       [0.08, 0.72, 0.20],
                       [0.02, 0.05, 0.93]]),
}

PRIMARY_SUCCESS_P = np.array([0.95, 0.50, 0.03])            # per health
# P(failure type | failure, health): timeout, 5xx, 429
FAIL_MIX = np.array([[0.5, 0.5, 0.0],
                     [0.5, 0.3, 0.2],
                     [0.7, 0.3, 0.0]])
SUCCESS_LAT = [(0.2, 0.35), (0.8, 0.60), (1.0, 0.60)]       # (median s, sigma)
FAIL_LAT = {TIMEOUT: 5.0, ERR5XX: 0.3, RATE429: 0.1}

PRIMARY_COST, FALLBACK_COST = 0.1, 2.5
FALLBACK_SUCCESS_P = 0.90
FALLBACK_LAT = (0.9, 0.4)
WAIT_DELAY = 2.0            # seconds spent waiting
WAIT_TICKS = 3              # health-chain ticks that pass during a WAIT
RATE_LIMIT_AFTER = 2        # >=2 immediate retries in a row -> 429 risk
RATE_LIMIT_P = 0.4
CACHE_MAX_AGE = 40.0        # seconds; staleness = age / CACHE_MAX_AGE
CACHE_PRESENT_P = 0.8
MAX_STEPS = 8


@dataclass
class RewardConfig:
    """Reward weights. Tune once on Saturday, then FREEZE."""
    r_success: float = 10.0       # reward for a delivered response
    alpha_latency: float = 0.5    # per second
    beta_cost: float = 1.0        # per cost unit
    gamma_stale: float = 5.0      # max staleness penalty on cache use
    delta_429: float = 3.0
    epsilon_fail: float = 10.0    # request abandoned with nothing
    cache_quality: float = 0.8    # cached answer is worth 80% of fresh primary
    fallback_quality: float = 0.85
    invalid_cache: float = 3.0    # tried USE_CACHE with empty cache (clearly costly)


def encode_state(last, fail, lat, cache, retry, fb):
    s = last
    s = s * N_FAIL + fail
    s = s * N_LAT + lat
    s = s * N_CACHE + cache
    s = s * N_RETRY + retry
    s = s * N_FB + fb
    return int(s)


def decode_state(s):
    """Inverse of encode_state -> (last, fail, lat, cache, retry, fb)."""
    s, fb = divmod(int(s), N_FB)
    s, retry = divmod(s, N_RETRY)
    s, cache = divmod(s, N_CACHE)
    s, lat = divmod(s, N_LAT)
    last, fail = divmod(s, N_FAIL)
    return last, fail, lat, cache, retry, fb


class RetryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, regime="normal", reward_cfg=None, seed=None):
        super().__init__()
        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.regime = regime
        self.P = REGIMES[regime]
        self.cfg = reward_cfg or RewardConfig()
        self.rng = np.random.default_rng(seed)
        self._init_world()

    # ------------------------------------------------------------ world
    def _init_world(self):
        self.health = HEALTHY
        self.fail_ema = 0.1
        self.lat_ema = 0.5
        for _ in range(50):                       # burn-in of background traffic
            self._tick(1)
            self._primary_call()

    def _tick(self, n=1):
        for _ in range(n):
            self.health = int(self.rng.choice(3, p=self.P[self.health]))

    def _lognormal(self, median, sigma):
        return float(self.rng.lognormal(np.log(median), sigma))

    def _primary_call(self, rate_pressure=False, force_fail=False):
        """Call the primary service. Returns (outcome, latency)."""
        h = self.health
        if rate_pressure and self.rng.random() < RATE_LIMIT_P:
            outcome = RATE429
        elif (not force_fail) and self.rng.random() < PRIMARY_SUCCESS_P[h]:
            outcome = SUCCESS
        else:
            outcome = int(self.rng.choice(3, p=FAIL_MIX[h]))
        lat = (self._lognormal(*SUCCESS_LAT[h]) if outcome == SUCCESS
               else FAIL_LAT[outcome])
        failed = float(outcome != SUCCESS)
        self.fail_ema = 0.8 * self.fail_ema + 0.2 * failed
        self.lat_ema = 0.7 * self.lat_ema + 0.3 * lat
        return outcome, lat

    # ----------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        if seed is not None:                      # re-seed AND rebuild the world
            self.rng = np.random.default_rng(seed)
            self._init_world()
        # background traffic between requests keeps the client statistics fresh
        for _ in range(3):
            self._tick(1)
            self._primary_call()
        # the request that triggers this episode: a FAILED first call
        self._tick(1)
        outcome, lat = self._primary_call(force_fail=True)

        self.last_outcome = outcome
        self.n_fail = 1                           # failed primary calls so far
        self.fb_tried = 0
        self.consec_retries = 0
        self.steps = 0
        self.cache_age = (float(self.rng.uniform(0, CACHE_MAX_AGE))
                          if self.rng.random() < CACHE_PRESENT_P else None)
        self.ep_latency = lat
        self.ep_cost = PRIMARY_COST
        self.ep_reward = 0.0
        self.ep_actions = np.zeros(N_ACTIONS, dtype=int)
        self._age_cache(lat)
        return self._obs(), {"health": self.health}

    def _age_cache(self, dt):
        if self.cache_age is not None:
            self.cache_age += dt

    def _bucket(self, x, cuts):
        return int(sum(x >= c for c in cuts))

    def _obs(self):
        fail = self._bucket(self.fail_ema, (0.50, 0.80))
        lat = self._bucket(self.lat_ema, (1.50, 3.00))
        if self.cache_age is None:
            cache = 0
        else:
            cache = 1 + self._bucket(self.cache_age, (8.0, 20.0))
        retry = min(self.n_fail, N_RETRY) - 1
        return encode_state(self.last_outcome, fail, lat, cache, retry, self.fb_tried)

    def step(self, action):
        c = self.cfg
        self.steps += 1
        self.ep_actions[action] += 1
        reward, lat, cost = 0.0, 0.0, 0.0
        terminated, served, source = False, False, None

        if action == USE_CACHE:
            if self.cache_age is None:
                reward -= c.invalid_cache
                lat = 0.1
            else:
                lat = 0.05
                stale = min(1.0, self.cache_age / CACHE_MAX_AGE)
                reward += c.r_success * c.cache_quality - c.gamma_stale * stale
                terminated, served, source = True, True, "cache"

        elif action == FALLBACK:
            self._tick(1)
            self.consec_retries = 0
            self.fb_tried = 1
            cost = FALLBACK_COST
            if self.rng.random() < FALLBACK_SUCCESS_P:
                lat = self._lognormal(*FALLBACK_LAT)
                reward += c.r_success * c.fallback_quality
                terminated, served, source = True, True, "fallback"
            else:
                lat = 2.0
                self.last_outcome = TIMEOUT

        else:  # RETRY or WAIT -> primary call
            pressure = False
            if action == WAIT:
                self._tick(WAIT_TICKS)
                lat += WAIT_DELAY
                self.consec_retries = 0
            else:
                self._tick(1)
                pressure = self.consec_retries >= RATE_LIMIT_AFTER
                self.consec_retries += 1
            outcome, call_lat = self._primary_call(rate_pressure=pressure)
            lat += call_lat
            cost = PRIMARY_COST
            if outcome == SUCCESS:
                reward += c.r_success
                terminated, served, source = True, True, "primary"
            else:
                self.n_fail += 1
                self.last_outcome = outcome
                if outcome == RATE429:
                    reward -= c.delta_429

        reward -= c.alpha_latency * lat + c.beta_cost * cost
        self.ep_latency += lat
        self.ep_cost += cost
        self._age_cache(lat)

        if not terminated and self.steps >= MAX_STEPS:
            reward -= c.epsilon_fail              # request abandoned
            terminated = True

        self.ep_reward += reward
        info = {"health": self.health}
        if terminated:
            info["episode"] = {
                "reward": self.ep_reward, "served": served, "source": source,
                "latency": self.ep_latency, "cost": self.ep_cost,
                "steps": self.steps, "actions": self.ep_actions.copy(),
            }
        return self._obs(), float(reward), terminated, False, info
