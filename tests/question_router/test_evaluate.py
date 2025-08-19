import csv
import json
import re
import logging

import pytest

from govuk_chat_evaluation.question_router.evaluate import (
    AggregateResults,
    EvaluationResult,
    evaluate_and_output_results,
)


class TestEvaluationResult:
    def test_for_csv(self):
        result = EvaluationResult(
            question="Test question",
            expected_outcome="genuine_rag",
            actual_outcome="greetings",
            confidence_score=0.9,
            answer=None,
        )

        assert result.for_csv() == {
            "question": "Test question",
            "expected_outcome": "genuine_rag",
            "actual_outcome": "greetings",
            "confidence_score": 0.9,
            "answer": None,
        }


class TestAggregateResults:
    @pytest.fixture
    def sample_results(self) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                question="Q1",
                expected_outcome="genuine_rag",
                actual_outcome="genuine_rag",
                confidence_score=0.95,
                answer=None,
            ),
            EvaluationResult(
                question="Q2",
                expected_outcome="about_mps",
                actual_outcome="about_mps",
                confidence_score=0.9,
                answer="This is an about mps answer",
            ),
            EvaluationResult(
                question="Q3",
                expected_outcome="character_fun",
                actual_outcome="genuine_rag",
                confidence_score=0.5,
                answer="This is a character fun answer",
            ),
            EvaluationResult(
                question="Q4",
                expected_outcome="character_fun",
                actual_outcome="about_mps",
                confidence_score=0.3,
                answer="This is a character fun answer",
            ),
            EvaluationResult(
                question="Q5",
                expected_outcome="genuine_rag",
                actual_outcome="genuine_rag",
                confidence_score=0.5,
                answer=None,
            ),
        ]

    def test_classification_labels(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert aggregate.classification_labels == [
            "about_mps",
            "character_fun",
            "genuine_rag",
        ]

    def test_accuracy_value(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert round(aggregate.accuracy(), 2) == 0.6

    def test_precision_value(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert round(aggregate.precision(), 2) == 0.61

    def test_recall_value(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert aggregate.recall() == 0.6

    def test_f1_value(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert round(aggregate.f1_score(), 2) == 0.45

    def test_f2_value(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert round(aggregate.f2_score(), 2) == 0.53

    def test_miscategorised_cases(self, sample_results):
        aggregate = AggregateResults(sample_results)
        miscategorised = aggregate.miscategorised_cases()
        logging.info(miscategorised)

        assert len(miscategorised) == 2
        assert miscategorised[0] == {
            "question": "Q3",
            "predicted_classification": "character_fun",
            "actual_classification": "genuine_rag",
            "confidence_score": 0.5,
            "answer": "This is a character fun answer",
        }
        assert miscategorised[1] == {
            "question": "Q4",
            "predicted_classification": "character_fun",
            "actual_classification": "about_mps",
            "confidence_score": 0.3,
            "answer": "This is a character fun answer",
        }

    def test_to_dict(self, sample_results):
        aggregate = AggregateResults(sample_results)
        assert aggregate.to_dict() == {
            "Evaluated": len(sample_results),
            "Accuracy": aggregate.accuracy(),
            "Precision": aggregate.precision(),
            "Recall": aggregate.recall(),
            "F1 Score": aggregate.f1_score(),
            "F2 Score": aggregate.f2_score(),
            "Miscategorised Cases": len(aggregate.miscategorised_cases()),
        }

    def test_for_csv(self, sample_results):
        aggregate = AggregateResults(sample_results)
        expected_csv = [
            {"property": k, "value": v} for k, v in aggregate.to_dict().items()
        ]
        assert aggregate.for_csv() == expected_csv


@pytest.fixture
def mock_evaluation_data_file(tmp_path):
    file_path = tmp_path / "evaluation_data.jsonl"
    data = [
        {
            "question": "Question 1",
            "expected_outcome": "genuine_rag",
            "actual_outcome": "genuine_rag",
            "confidence_score": 0.95,
            "answer": None,
        },
        {
            "question": "Question 2",
            "expected_outcome": "genuine_rag",
            "actual_outcome": "about_mps",
            "confidence_score": 0.95,
            "answer": "This is an about mps answer",
        },
    ]

    with open(tmp_path / "evaluation_data.jsonl", "w", encoding="utf8") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")

    return file_path


def test_evaluate_and_output_results_writes_results(
    mock_project_root, mock_evaluation_data_file
):
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)
    results_file = mock_project_root / "results.csv"

    assert results_file.exists()

    with open(results_file, "r") as file:
        reader = csv.reader(file)
        headers = next(reader, None)

        assert headers is not None
        assert "question" in headers


def test_evaluate_and_output_results_writes_aggregates(
    mock_project_root, mock_evaluation_data_file
):
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)
    aggregates_file = mock_project_root / "aggregate.csv"

    assert aggregates_file.exists()
    with open(aggregates_file, "r") as file:
        reader = csv.reader(file)
        headers = next(reader, None)

        assert headers is not None
        assert "property" in headers


def test_evaluate_and_output_results_writes_confusion_matrix(
    mock_project_root, mock_evaluation_data_file
):
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)
    confusion_matrix_file = mock_project_root / "confusion_matrix.png"

    assert confusion_matrix_file.exists()


def test_evaluate_and_output_results_writes_miscategorised_cases(
    mock_project_root, mock_evaluation_data_file
):
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)
    miscategorised_cases_file = mock_project_root / "miscategorised_cases.csv"

    assert miscategorised_cases_file.exists()
    with open(miscategorised_cases_file, "r") as file:
        reader = csv.reader(file)
        headers = next(reader, None)

        assert headers is not None
        assert "question" in headers
        assert "predicted_classification" in headers
        assert "actual_classification" in headers
        assert "answer" in headers


def test_evaluate_and_output_results_prints_aggregates(
    mock_project_root, mock_evaluation_data_file, caplog
):
    caplog.set_level(logging.INFO)
    evaluate_and_output_results(mock_project_root, mock_evaluation_data_file)

    assert "Aggregate Results" in caplog.text
    assert re.search(r"Evaluated\s+\d+", caplog.text)


def test_evaluate_and_output_results_copes_with_empty_data(
    mock_project_root, tmp_path, caplog
):
    caplog.set_level(logging.ERROR)
    file_path = tmp_path / "evaluation_data.jsonl"
    file_path.touch()

    evaluate_and_output_results(mock_project_root, file_path)

    assert "There is no data to evaluate" in caplog.text
