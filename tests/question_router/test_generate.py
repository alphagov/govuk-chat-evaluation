import json
from unittest.mock import AsyncMock

import pytest

from govuk_chat_evaluation.question_router.generate import (
    generate_inputs_to_evaluation_results,
    generate_and_write_dataset,
    GenerateInput,
    EvaluationResult,
)


@pytest.fixture
def run_rake_task_mock(mocker):
    async def default_side_effect(_, env):
        if env["INPUT"] == "Question 1":
            return {
                "question_routing_label": "genuine_rag",
                "question_routing_confidence_score": 0.9,
                "message": None,
                "metrics": {"question_routing": {"model": "model_name"}},
            }
        else:
            return {
                "question_routing_label": "greetings",
                "question_routing_confidence_score": 0.8,
                "message": "This is a greetings answer",
                "metrics": {"question_routing": {"model": "model_name"}},
            }

    mock = mocker.patch(
        "govuk_chat_evaluation.question_router.generate.run_rake_task",
        new_callable=AsyncMock,
    )
    mock.side_effect = default_side_effect
    return mock


def test_generate_inputs_to_evaluation_results_returns_evaluation_results(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            question="Question 1",
            expected_outcome="genuine_rag",
        ),
        GenerateInput(
            question="Question 2",
            expected_outcome="greetings",
        ),
    ]
    expected_results = [
        EvaluationResult(
            question="Question 1",
            expected_outcome="genuine_rag",
            actual_outcome="genuine_rag",
            confidence_score=0.9,
            answer=None,
            model="model_name",
        ),
        EvaluationResult(
            question="Question 2",
            expected_outcome="greetings",
            actual_outcome="greetings",
            confidence_score=0.8,
            answer="This is a greetings answer",
            model="model_name",
        ),
    ]
    actual_results = generate_inputs_to_evaluation_results(None, generate_inputs)

    assert sorted(expected_results, key=lambda r: r.question) == sorted(
        actual_results, key=lambda r: r.question
    )


def test_generate_inputs_to_evaluation_results_runs_expected_rake_task(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            question="Question 1",
            expected_outcome="genuine_rag",
        ),
    ]
    generate_inputs_to_evaluation_results(None, generate_inputs)

    run_rake_task_mock.assert_called_with(
        "evaluation:generate_question_routing_response",
        {"INPUT": "Question 1"},
    )


def test_generate_models_with_claude_generation_model_populates_model_env_var_for_rake_task(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            question="Question 1",
            expected_outcome="genuine_rag",
        ),
    ]
    generate_inputs_to_evaluation_results("claude_sonnet_4_0", generate_inputs)

    run_rake_task_mock.assert_called_with(
        "evaluation:generate_question_routing_response",
        {
            "INPUT": "Question 1",
            "BEDROCK_CLAUDE_QUESTION_ROUTER_MODEL": "claude_sonnet_4_0",
        },
    )


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_and_write_dataset(mock_input_data, mock_project_root):
    path = generate_and_write_dataset(mock_input_data, None, mock_project_root)
    assert path.exists()
    with open(path, "r") as file:
        for line in file:
            assert json.loads(line)
