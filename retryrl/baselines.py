"""
Rule-based comparison policies. These are BENCHMARKS ONLY (not the solution).
Each policy maps a discrete state index -> action.
"""
import numpy as np

from .env import RETRY, WAIT, USE_CACHE, FALLBACK, decode_state


def always_retry(s, rng=None):
    return RETRY


def exp_backoff(s, rng=None):
    # wait-and-retry for the first 3 failures, then switch to fallback
    last, fail, lat, cache, retry, fb = decode_state(s)
    return WAIT if retry < 3 and not fb else FALLBACK


def cache_first(s, rng=None):
    # use the cache if there is one; otherwise retry, then fallback
    last, fail, lat, cache, retry, fb = decode_state(s)
    if cache > 0:
        return USE_CACHE
    return RETRY if retry < 3 and not fb else FALLBACK


def threshold_fallback(s, rng=None):
    # retry twice, then go to the fallback and stay there
    last, fail, lat, cache, retry, fb = decode_state(s)
    return RETRY if retry < 2 and not fb else FALLBACK


def random_policy(s, rng=None):
    rng = rng or np.random.default_rng()
    return int(rng.integers(4))


BASELINES = {
    "AlwaysRetry": always_retry,
    "ExpBackoff": exp_backoff,
    "CacheFirst": cache_first,
    "ThresholdFallback": threshold_fallback,
    "Random": random_policy,
}
