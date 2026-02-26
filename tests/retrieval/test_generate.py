import json
from unittest.mock import AsyncMock

import pytest

from govuk_chat_evaluation.retrieval.generate import (
    generate_inputs_to_evaluation_results,
    generate_and_write_dataset,
    GenerateInput,
    EvaluationResult,
)


@pytest.fixture
def run_rake_task_mock(mocker):
    async def default_side_effect(_, env):
        if env["INPUT"] == "Question 1":
            return [
                {
                    "exact_path": "/foo",
                    "chunk_uid": "uid1",
                    "plain_content": "Content for foo",
                    "weighted_score": 1.0,
                    "original_score": 1.5,
                },
                {
                    "exact_path": "/bar",
                    "chunk_uid": "uid2",
                    "plain_content": "Content for bar",
                    "weighted_score": 0.8,
                    "original_score": 0.9,
                },
                {
                    "exact_path": "/baz",
                    "chunk_uid": "uid3",
                    "plain_content": "Content for baz",
                    "weighted_score": 0.5,
                    "original_score": 0.4,
                },
            ]
        else:
            return [
                {
                    "exact_path": "/path1",
                    "chunk_uid": "uid4",
                    "plain_content": "Content for path1",
                    "weighted_score": 1.0,
                    "original_score": 1.1,
                },
                {
                    "exact_path": "/path2",
                    "chunk_uid": "uid5",
                    "plain_content": "Content for path2",
                    "weighted_score": 0.9,
                    "original_score": 0.8,
                },
            ]

    mock = mocker.patch(
        "govuk_chat_evaluation.retrieval.generate.run_rake_task",
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
            expected_exact_paths=["/foo"],
            expected_chunk_uids=["uid1"]

        ),
        GenerateInput(
            question="Question 2",
            expected_exact_paths=["/path1", "/path2"],
            expected_chunk_uids=["uid4", "uid5"]
        ),
    ]
    expected_results = [
        EvaluationResult(
            question="Question 1",
            expected_exact_paths=["/foo"],
            expected_chunk_uids=["uid1"],
            actual_chunk_uids_exact_paths_and_scores=[
                {"exact_path": "/foo", "chunk_uid": "uid1", "weighted_score": 1.0, "original_score": 1.5},
                {"exact_path": "/bar", "chunk_uid": "uid2", "weighted_score": 0.8, "original_score": 0.9},
                {"exact_path": "/baz", "chunk_uid": "uid3", "weighted_score": 0.5, "original_score": 0.4},
            ],
        ),
        EvaluationResult(
            question="Question 2",
            expected_exact_paths=["/path1", "/path2"],
            expected_chunk_uids=["uid4", "uid5"],
            actual_chunk_uids_exact_paths_and_scores=[
                {"exact_path": "/path1", "chunk_uid": "uid4", "weighted_score": 1.0, "original_score": 1.1},
                {"exact_path": "/path2", "chunk_uid": "uid5", "weighted_score": 0.9, "original_score": 0.8},
            ],
        ),
    ]
    actual_results = generate_inputs_to_evaluation_results("titan", generate_inputs)

    assert sorted(expected_results, key=lambda r: r.question) == sorted(
        actual_results, key=lambda r: r.question
    )


def test_generate_inputs_to_evaluation_results_runs_expected_rake_task(
    run_rake_task_mock,
):
    generate_inputs = [
        GenerateInput(
            question="Question 1",
            expected_exact_paths=["/foo", "/bar"],
            expected_chunk_uids=["uid1", "uid2"],
        ),
    ]

    generate_inputs_to_evaluation_results("titan", generate_inputs)

    run_rake_task_mock.assert_called_with(
        "evaluation:search_results_for_question",
        {"INPUT": "Question 1", "EMBEDDING_PROVIDER": "titan"},
    )


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_and_write_dataset(mock_input_data, mock_project_root):
    path = generate_and_write_dataset(mock_input_data, "titan", mock_project_root)
    assert path.exists()
    with open(path, "r") as file:
        for line in file:
            assert json.loads(line)
