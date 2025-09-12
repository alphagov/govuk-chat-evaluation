import pytest
import csv
import json
import re
import logging

from govuk_chat_evaluation.topic_tagger.evaluate import (
    AggregateResults,
    EvaluationResult,
    evaluate_and_output_results,
)


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    return [
        EvaluationResult(
            question="Q1",
            expected_primary_topic="benefits",
            actual_primary_topic="benefits",
            expected_secondary_topic="tax",
            actual_secondary_topic="tax",
        ),
        EvaluationResult(
            question="Q2",
            expected_primary_topic="benefits",
            actual_primary_topic="tax",
            expected_secondary_topic="tax",
            actual_secondary_topic="benefits",
        ),
        EvaluationResult(
            question="Q3",
            expected_primary_topic="benefits",
            actual_primary_topic="benefits",
            expected_secondary_topic="tax",
            actual_secondary_topic=None,
        ),
        EvaluationResult(
            question="Q4",
            expected_primary_topic="tax",
            actual_primary_topic="childcare",
            expected_secondary_topic="benefits",
            actual_secondary_topic="tax",
        ),
        EvaluationResult(
            question="Q5",
            expected_primary_topic="benefits",
            actual_primary_topic="tax",
            expected_secondary_topic="tax",
            actual_secondary_topic=None,
        ),
        EvaluationResult(
            question="Q6",
            expected_primary_topic="benefits",
            actual_primary_topic="tax",
            expected_secondary_topic=None,
            actual_secondary_topic="driving",
        ),
    ]


class TestEvaluationResult:
    @pytest.mark.parametrize("index", [0])
    def test_correct_primary_and_secondary_match(self, sample_results, index):
        assert sample_results[index].correct_primary_and_secondary()

    @pytest.mark.parametrize("index", range(1, 5))
    def test_correct_primary_and_secondary_no_match(self, sample_results, index):
        assert not sample_results[index].correct_primary_and_secondary()

    @pytest.mark.parametrize("index", range(0, 1))
    def test_correct_topics_any_order_match(self, sample_results, index):
        assert sample_results[index].correct_topics_any_order()

    @pytest.mark.parametrize("index", range(2, 5))
    def test_correct_topics_any_order_no_match(self, sample_results, index):
        assert not sample_results[index].correct_topics_any_order()

    @pytest.mark.parametrize("index", [0, 2])
    def test_match_true_primary_with_primary_match(self, sample_results, index):
        assert sample_results[index].matched_true_primary_with_primary()

    @pytest.mark.parametrize("index", [1, 3, 4, 5])
    def test_match_true_primary_with_primary_no_match(self, sample_results, index):
        assert not sample_results[index].matched_true_primary_with_primary()

    @pytest.mark.parametrize("index", range(0, 3))
    def test_true_primary_with_either_match(self, sample_results, index):
        assert sample_results[index].matched_true_primary_with_either()

    @pytest.mark.parametrize("index", range(4, 5))
    def test_true_primary_with_either_no_match(self, sample_results, index):
        assert not sample_results[index].matched_true_primary_with_either()

    @pytest.mark.parametrize("index", range(0, 4))
    def test_matched_any_topic_match(self, sample_results, index):
        assert sample_results[index].matched_any_topic()

    @pytest.mark.parametrize("index", [5])
    def test_matched_any_topic_no_match(self, sample_results, index):
        assert not sample_results[index].matched_any_topic()

    def test_matched_any_topic_with_no_secondary_returns_false(self):
        result = EvaluationResult(
            question="Q1",
            expected_primary_topic="benefits",
            actual_primary_topic="tax",
            expected_secondary_topic=None,
            actual_secondary_topic=None,
        )
        assert not result.matched_any_topic()

    def test_for_csv(self, sample_results):
        csv_row = sample_results[0].for_csv()
        assert csv_row["correct_primary_and_secondary"] is True
        assert csv_row["correct_topics_any_order"] is True
        assert csv_row["matched_true_primary_with_primary"] is True
        assert csv_row["matched_true_primary_with_either"] is True


class TestAggregateResults:
    def test_aggregate_metrics(self, sample_results):
        aggregate = AggregateResults(sample_results)
        result = aggregate.to_dict()

        assert result["Evaluated"] == 6
        assert result["Correct Primary and Secondary"] == 1
        assert result["Correct Topics (any order)"] == 2
        assert result["Matched True primary with primary"] == 2
        assert result["Matched True primary with either topic"] == 4
        assert result["Matched Any Topic"] == 5

    def test_for_csv(self, sample_results):
        aggregate = AggregateResults(sample_results)
        expected_keys = {
            "Evaluated",
            "Correct Primary and Secondary",
            "Correct Topics (any order)",
            "Matched True primary with primary",
            "Matched True primary with either topic",
            "Matched Any Topic",
        }
        csv_rows = aggregate.for_csv()
        assert {row["property"] for row in csv_rows} == expected_keys


@pytest.fixture
def mock_evaluation_data_file(tmp_path, sample_results):
    file_path = tmp_path / "evaluation_data.jsonl"
    payloads = [item.model_dump() for item in sample_results]

    with open(file_path, "w", encoding="utf8") as file:
        for payload in payloads:
            file.write(json.dumps(payload) + "\n")

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
