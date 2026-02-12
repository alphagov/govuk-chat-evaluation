import json
from unittest.mock import AsyncMock

import pytest

from govuk_chat_evaluation.output_guardrails.generate import (
    generate_inputs_to_evaluation_results,
    generate_and_write_dataset,
    GenerateInput,
)

from govuk_chat_evaluation.output_guardrails.evaluate import EvaluationResult


@pytest.fixture
def run_rake_task_mock(mocker):
    async def default_side_effect(_, env):
        if env["INPUT"] == "This answer contains inappropriate content.":
            return {
                "triggered": True,
                "answer_guardrails_failures": ["appropriate_language", "political"],
                "metrics": {"answer_guardrails": {"model": "model_name"}},
            }
        else:
            return {
                "triggered": False,
                "answer_guardrails_failures": [],
                "metrics": {"answer_guardrails": {"model": "model_name"}},
            }

    mock = mocker.patch(
        "govuk_chat_evaluation.output_guardrails.generate.run_rake_task",
        new_callable=AsyncMock,
    )
    mock.side_effect = default_side_effect
    return mock


def test_generate_inputs_to_evaluation_results_returns_evaluation_results(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            message="This answer contains inappropriate content.",
            expected_triggered=True,
            expected_guardrails={
                "appropriate_language": True,
                "political": True,
                "contains_pii": False,
            },
        ),
        GenerateInput(
            message="This is a safe and appropriate answer.",
            expected_triggered=False,
            expected_guardrails={
                "appropriate_language": False,
                "political": False,
            },
        ),
    ]
    expected_results = [
        EvaluationResult(
            message="This answer contains inappropriate content.",
            expected_triggered=True,
            actual_triggered=True,
            expected_guardrails={
                "appropriate_language": True,
                "political": True,
                "contains_pii": False,
            },
            actual_guardrails={
                "appropriate_language": True,
                "political": True,
                "contains_pii": False,
            },
            model="model_name",
        ),
        EvaluationResult(
            message="This is a safe and appropriate answer.",
            expected_triggered=False,
            actual_triggered=False,
            expected_guardrails={
                "appropriate_language": False,
                "political": False,
            },
            actual_guardrails={
                "appropriate_language": False,
                "political": False,
            },
            model="model_name",
        ),
    ]
    actual_results = generate_inputs_to_evaluation_results(
        "answer_guardrails", None, generate_inputs
    )

    assert sorted(expected_results, key=lambda r: r.message) == sorted(
        actual_results, key=lambda r: r.message
    )


def test_generate_inputs_to_evaluation_results_runs_expected_rake_task(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            message="This answer contains inappropriate content.",
            expected_triggered=True,
            expected_guardrails={"appropriate_language": True},
        ),
    ]
    generate_inputs_to_evaluation_results("answer_guardrails", None, generate_inputs)

    run_rake_task_mock.assert_called_with(
        "evaluation:generate_output_guardrail_response[answer_guardrails]",
        {"INPUT": "This answer contains inappropriate content."},
    )


def test_generate_models_with_claude_generation_model_populates_model_env_var_for_rake_task(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            message="This answer contains inappropriate content.",
            expected_triggered=True,
            expected_guardrails={"appropriate_language": True},
        ),
    ]
    generate_inputs_to_evaluation_results(
        "answer_guardrails", "claude_sonnet_4_0", generate_inputs
    )

    run_rake_task_mock.assert_called_with(
        "evaluation:generate_output_guardrail_response[answer_guardrails]",
        {
            "INPUT": "This answer contains inappropriate content.",
            "BEDROCK_CLAUDE_GUARDRAILS_MODEL": "claude_sonnet_4_0",
        },
    )


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_and_write_dataset(mock_input_data, mock_project_root):
    path = generate_and_write_dataset(
        mock_input_data, "answer_guardrails", None, mock_project_root
    )
    assert path.exists()
    with open(path, "r") as file:
        for line in file:
            assert json.loads(line)
