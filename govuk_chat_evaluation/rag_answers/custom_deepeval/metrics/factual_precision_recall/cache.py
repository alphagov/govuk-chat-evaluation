from typing import Tuple
from .schema import ClassifiedFacts

CacheKey = Tuple[str, str, str]


class FactClassificationCache:
    def __init__(self):
        self._store: dict[CacheKey, ClassifiedFacts] = {}

    def _make_key(
        self, evaluation_model: str | None, answer: str, ground_truth: str
    ) -> CacheKey:
        return (evaluation_model or "unknown-model", answer, ground_truth)

    def get(
        self, evaluation_model: str | None, answer: str, ground_truth: str
    ) -> ClassifiedFacts | None:
        key = self._make_key(evaluation_model, answer, ground_truth)
        return self._store.get(key)

    def set(
        self,
        evaluation_model: str | None,
        answer: str,
        ground_truth: str,
        value: ClassifiedFacts,
    ) -> None:
        key = self._make_key(evaluation_model, answer, ground_truth)
        self._store[key] = value
