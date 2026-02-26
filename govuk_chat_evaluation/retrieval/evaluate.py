from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)
from tabulate import tabulate
import numpy as np
from collections.abc import Callable

from ..file_system import jsonl_to_models, write_csv_results
import logging

DECIMAL_PLACES = 4


class ChunkScores(BaseModel):
    exact_path: str
    chunk_uid: str
    weighted_score: float
    original_score: float


class EvaluationResult(BaseModel):
    question: str
    expected_exact_paths: list[str]
    expected_chunk_uids: list[str]
    actual_chunk_uids_exact_paths_and_scores: list[ChunkScores]

    @property
    def actual_chunk_uids(self) -> list[str]:
        return [
            item.chunk_uid for item in self.actual_chunk_uids_exact_paths_and_scores
        ]

    @property
    def all_chunk_uids(self) -> list[str]:
        return list(set(self.expected_chunk_uids + self.actual_chunk_uids))

    @property
    def y_true(self) -> list[int]:
        return [
            int(chunk_uid in self.expected_chunk_uids)
            for chunk_uid in self.all_chunk_uids
        ]

    @property
    def y_pred(self) -> list[int]:
        return [
            int(chunk_uid in self.actual_chunk_uids)
            for chunk_uid in self.all_chunk_uids
        ]

    def _safe_classification_metric(
        self, metric_fn: Callable[..., float], **kwargs: Any
    ) -> float:
        if not self.all_chunk_uids:
            return float("nan")
        return metric_fn(self.y_true, self.y_pred, **kwargs)

    @property
    def false_positive_cases(self) -> list[dict[str, float]]:
        return [
            {item.exact_path: item.weighted_score}
            for item in self.actual_chunk_uids_exact_paths_and_scores
            if item.chunk_uid not in self.expected_chunk_uids
        ]

    @property
    def false_negative_cases(self) -> list[dict[str, float]]:
        return [
            {path: float("nan")}
            for path, uid in zip(self.expected_exact_paths, self.expected_chunk_uids)
            if uid not in self.actual_chunk_uids
        ]

    @property
    def true_positive_cases(self) -> list[dict[str, float]]:
        return [
            {item.exact_path: item.weighted_score}
            for item in self.actual_chunk_uids_exact_paths_and_scores
            if item.chunk_uid in self.expected_chunk_uids
        ]

    def precision(self) -> float:
        return self._safe_classification_metric(
            precision_score,
            zero_division=np.nan,  # type: ignore
        )

    def recall(self) -> float:
        return self._safe_classification_metric(
            recall_score,
            zero_division=np.nan,  # type: ignore
        )

    def f1_score(self) -> float:
        return self._safe_classification_metric(
            f1_score,
            zero_division=np.nan,  # type: ignore
        )

    def f2_score(self) -> float:
        return self._safe_classification_metric(
            fbeta_score,
            beta=2,
            zero_division=np.nan,  # type: ignore
        )

    def for_csv(self) -> dict[str, Any]:
        tuples = []
        for item in self.actual_chunk_uids_exact_paths_and_scores:
            tuples.append((item.exact_path, item.chunk_uid, item.weighted_score))
        return {
            "question": self.question,
            "expected_exact_paths": self.expected_exact_paths,
            "expected_chunk_uids": self.expected_chunk_uids,
            "actual_chunk_uids_exact_paths_and_scores": tuples,
            "precision": round(self.precision(), DECIMAL_PLACES),
            "recall": round(self.recall(), DECIMAL_PLACES),
            "f1_score": round(self.f1_score(), DECIMAL_PLACES),
            "f2_score": round(self.f2_score(), DECIMAL_PLACES),
            "true_positives": self.true_positive_cases,
            "false_negatives": self.false_negative_cases,
            "false_positives": self.false_positive_cases,
        }


class AggregateResults:
    def __init__(self, evaluation_results: list[EvaluationResult]):
        self.evaluation_results = evaluation_results

    def _aggregate(
        self,
        score_fn: Callable[[Any], float],
        agg_fn: Callable[[list[float]], float],
    ) -> float:
        scores = [score_fn(result) for result in self.evaluation_results]
        result = agg_fn(scores)
        return float(round(result, DECIMAL_PLACES))

    def precision_mean(self) -> float:
        return self._aggregate(lambda r: r.precision(), np.mean)

    def precision_median(self) -> float:
        return self._aggregate(lambda r: r.precision(), np.median)

    def precision_max(self) -> float:
        return self._aggregate(lambda r: r.precision(), np.max)

    def precision_standard_deviation(self) -> float:
        return self._aggregate(lambda r: r.precision(), np.std)

    def recall_mean(self) -> float:
        return self._aggregate(lambda r: r.recall(), np.mean)

    def recall_median(self) -> float:
        return self._aggregate(lambda r: r.recall(), np.median)

    def recall_max(self) -> float:
        return self._aggregate(lambda r: r.recall(), np.max)

    def recall_standard_deviation(self) -> float:
        return self._aggregate(lambda r: r.recall(), np.std)

    def f1_mean(self) -> float:
        return self._aggregate(lambda r: r.f1_score(), np.mean)

    def f1_median(self) -> float:
        return self._aggregate(lambda r: r.f1_score(), np.median)

    def f1_max(self) -> float:
        return self._aggregate(lambda r: r.f1_score(), np.max)

    def f1_standard_deviation(self) -> float:
        return self._aggregate(lambda r: r.f1_score(), np.std)

    def f2_mean(self) -> float:
        return self._aggregate(lambda r: r.f2_score(), np.mean)

    def f2_median(self) -> float:
        return self._aggregate(lambda r: r.f2_score(), np.median)

    def f2_max(self) -> float:
        return self._aggregate(lambda r: r.f2_score(), np.max)

    def f2_standard_deviation(self) -> float:
        return self._aggregate(lambda r: r.f2_score(), np.std)

    def to_dict(self) -> dict[str, Any]:
        return {
            "Evaluated": len(self.evaluation_results),
            "Precision mean": self.precision_mean(),
            "Precision median": self.precision_median(),
            "Precision max": self.precision_max(),
            "Precision standard deviation": self.precision_standard_deviation(),
            "Recall mean": self.recall_mean(),
            "Recall median": self.recall_median(),
            "Recall max": self.recall_max(),
            "Recall standard deviation": self.recall_standard_deviation(),
            "F1 mean": self.f1_mean(),
            "F1 median": self.f1_median(),
            "F1 max": self.f1_max(),
            "F1 standard deviation": self.f1_standard_deviation(),
            "F2 mean": self.f2_mean(),
            "F2 median": self.f2_median(),
            "F2 max": self.f2_max(),
            "F2 standard deviation": self.f2_standard_deviation(),
        }

    def for_csv(self) -> list[dict[str, Any]]:
        return [{"property": k, "value": v} for k, v in self.to_dict().items()]


def evaluate_and_output_results(output_dir: Path, evaluation_data_path: Path):
    """Evaluate the data in the evaluation data file and write result files
    to the output paths, with aggregates written to STDOUT"""

    models = jsonl_to_models(evaluation_data_path, EvaluationResult)

    if not models:
        logging.error("\nThere is no data to evaluate")
        return

    logging.info("\nEvaluation complete")
    write_csv_results(output_dir, [model.for_csv() for model in models])

    aggregate_results = AggregateResults(models)

    write_csv_results(
        output_dir,
        aggregate_results.for_csv(),
        filename="aggregate.csv",
        data_label="aggregates",
    )

    table = [[k, v] for k, v in aggregate_results.to_dict().items()]
    logging.info("\nAggregate Results")
    logging.info(tabulate(table) + "\n")
