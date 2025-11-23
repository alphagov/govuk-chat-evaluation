from typing import Tuple

from .schema import ClassifiedFacts


CacheKey = Tuple[str, str, str]

# Process-local cache of fact classifications keyed by (evaluation_model, answer, ground_truth)
_CACHE: dict[CacheKey, ClassifiedFacts] = {}


def make_cache_key(
    evaluation_model: str | None, answer: str, ground_truth: str
) -> CacheKey:
    return (evaluation_model or "unknown-model", answer, ground_truth)


def get_cache() -> dict[CacheKey, ClassifiedFacts]:
    return _CACHE


def reset_cache() -> None:
    _CACHE.clear()
