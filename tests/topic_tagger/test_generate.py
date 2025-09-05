import pytest
import json
from unittest.mock import AsyncMock
from govuk_chat_evaluation.topic_tagger.generate import (
    generate_inputs_to_evaluation_results,
    generate_and_write_dataset,
    GenerateInput,
    EvaluationResult,
)


@pytest.fixture
def run_rake_task_mock(mocker):
    return mocker.patch(
        "govuk_chat_evaluation.topic_tagger.generate.run_rake_task",
        new_callable=AsyncMock,
        return_value={"primary_topic": "benefits", "secondary_topic": "tax"},
    )


def test_generate_models_to_evaluation_results_returns_evaluation_results(
    run_rake_task_mock, mock_data
):
    result_per_question = [
        {
            "primary_topic": item["actual_primary_topic"],
            "secondary_topic": item["actual_secondary_topic"],
        }
        for item in mock_data
    ]

    run_rake_task_mock.side_effect = result_per_question

    generate_inputs = [
        GenerateInput(
            question=item["question"],
            expected_primary_topic=item["expected_primary_topic"],
            expected_secondary_topic=item["expected_secondary_topic"],
        )
        for item in mock_data
    ]

    expected_results = [
        EvaluationResult(
            question=item["question"],
            expected_primary_topic=item["expected_primary_topic"],
            actual_primary_topic=item["actual_primary_topic"],
            expected_secondary_topic=item["expected_secondary_topic"],
            actual_secondary_topic=item["actual_secondary_topic"],
        )
        for item in mock_data
    ]

    actual_results = generate_inputs_to_evaluation_results(generate_inputs)

    assert sorted(expected_results, key=lambda r: r.question) == sorted(
        actual_results, key=lambda r: r.question
    )


def test_generate_models_to_evaluation_results_raises_on_unexpected_key(
    run_rake_task_mock,
):
    run_rake_task_mock.return_value = {"what": {"is": "this?"}}
    generate_inputs = [
        GenerateInput(
            question="Question 1",
            expected_primary_topic="benefits",
            expected_secondary_topic="tax",
        ),
    ]
    with pytest.raises(RuntimeError) as exc_info:
        generate_inputs_to_evaluation_results(generate_inputs)

    assert "Unexpected result structure {'what': {'is': 'this?'}}" in str(
        exc_info.value
    )


def test_generate_models_to_evaluation_models_runs_expected_rake_task(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            question="Question 1",
            expected_primary_topic="benefits",
            expected_secondary_topic="tax",
        ),
    ]
    generate_inputs_to_evaluation_results(generate_inputs)

    run_rake_task_mock.assert_called_with(
        "evaluation:generate_topics_for_question",
        {"INPUT": "Question 1"},
    )


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_and_write_dataset(mock_input_data, mock_project_root):
    path = generate_and_write_dataset(mock_input_data, mock_project_root)
    assert path.exists()
    with open(path, "r") as file:
        for line in file:
            parsed = json.loads(line)
            assert "question" in parsed
            assert "expected_primary_topic" in parsed
            assert "actual_primary_topic" in parsed
            assert "expected_secondary_topic" in parsed
            assert "actual_secondary_topic" in parsed
