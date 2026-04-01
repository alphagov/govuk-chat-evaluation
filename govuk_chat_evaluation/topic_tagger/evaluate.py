from pathlib import Path
from typing import Any
from enum import Enum

from pydantic import BaseModel
from tabulate import tabulate

from ..file_system import jsonl_to_models, write_csv_results
import logging


class TopicStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class EvaluationResult(BaseModel):
    question: str
    status: TopicStatus
    expected_primary_topic: str
    actual_primary_topic: str | None
    expected_secondary_topic: str | None
    actual_secondary_topic: str | None
    error_message: str | None = None
    model: str | None

    @property
    def expected_topics(self) -> set[str]:
        return {
            topics
            for topics in (self.expected_primary_topic, self.expected_secondary_topic)
            if topics is not None
        }

    @property
    def actual_topics(self) -> set[str]:
        return {
            topics
            for topics in (self.actual_primary_topic, self.actual_secondary_topic)
            if topics is not None
        }

    def correct_primary_and_secondary(self) -> bool:
        return (
            self.expected_primary_topic == self.actual_primary_topic
            and self.expected_secondary_topic == self.actual_secondary_topic
        )

    def correct_topics_any_order(self) -> bool:
        return self.expected_topics == self.actual_topics

    def matched_true_primary_with_primary(self) -> bool:
        return self.expected_primary_topic == self.actual_primary_topic

    def matched_true_primary_with_either(self) -> bool:
        return self.expected_primary_topic in self.actual_topics

    def matched_any_topic(self) -> bool:
        return bool(self.expected_topics & self.actual_topics)

    def for_csv(self) -> dict[str, Any]:
        return {
            **self.model_dump(),
            "correct_primary_and_secondary": self.correct_primary_and_secondary(),
            "correct_topics_any_order": self.correct_topics_any_order(),
            "matched_true_primary_with_primary": self.matched_true_primary_with_primary(),
            "matched_true_primary_with_either": self.matched_true_primary_with_either(),
        }


class AggregateResults:
    def __init__(self, evaluation_results: list[EvaluationResult]):
        self.success_evaluation_results = [
            r for r in evaluation_results if r.status == TopicStatus.SUCCESS
        ]
        self.error_evaluation_results = [
            r for r in evaluation_results if r.status == TopicStatus.ERROR
        ]
        self.model = (
            self.success_evaluation_results[0].model
            if self.success_evaluation_results
            else None
        )
        self.correct_primary_and_secondary = sum(
            r.correct_primary_and_secondary() for r in self.success_evaluation_results
        )
        self.correct_topics_any_order = sum(
            r.correct_topics_any_order() for r in self.success_evaluation_results
        )
        self.matched_true_primary_with_primary = sum(
            r.matched_true_primary_with_primary()
            for r in self.success_evaluation_results
        )
        self.matched_true_primary_with_either = sum(
            r.matched_true_primary_with_either()
            for r in self.success_evaluation_results
        )
        self.matched_any_topic = sum(
            r.matched_any_topic() for r in self.success_evaluation_results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "Evaluated": len(self.success_evaluation_results),
            "Errored": len(self.error_evaluation_results),
            "Model": self.model,
            "Correct Primary and Secondary": self.correct_primary_and_secondary,
            "Correct Topics (any order)": self.correct_topics_any_order,
            "Matched True primary with primary": self.matched_true_primary_with_primary,
            "Matched True primary with either topic": self.matched_true_primary_with_either,
            "Matched Any Topic": self.matched_any_topic,
        }

    def for_csv(self) -> list[dict[str, Any]]:
        return [{"property": k, "value": v} for k, v in self.to_dict().items()]


def evaluate_and_output_results(output_dir: Path, evaluation_data_path: Path):
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
