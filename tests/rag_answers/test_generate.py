import json
from unittest.mock import AsyncMock

import pytest

from govuk_chat_evaluation.rag_answers.generate import (
    generate_inputs_to_evaluation_test_cases,
    generate_and_write_dataset,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    GenerateInput,
    EvaluationTestCase,
    StructuredContext,
)


@pytest.fixture
def run_rake_task_mock(mocker):
    mock = mocker.patch(
        "govuk_chat_evaluation.rag_answers.generate.run_rake_task",
        new_callable=AsyncMock,
    )

    sources = [
        {
            "used": True,
            "search_score": 0.1423,
            "weighted_score": 1.51231,
            "chunk": {
                "content_id": "383f0b50-e205-4695-8204-1d8b4ff19346",
                "locale": "en",
                "chunk_index": 0,
                "digest": "feca7f0d1bddfde25b3627e72716f2fb91441bf8498d48d11fcb67fcc2186015",
                "title": "Title",
                "description": None,
                "heading_hierarchy": ["Heading 1", "Heading 2"],
                "base_path": "/income-tax",
                "exact_path": "/income-tax",
                "document_type": "guide",
                "parent_document_type": None,
                "html_content": "<p>Some content</p>",
                "plain_content": "Some content",
            },
        }
    ]
    mock.side_effect = lambda *_: {"message": "An answer", "sources": sources}
    return mock


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_models_to_evaluation_test_cases_returns_evaluation_test_cases():
    generate_inputs = [
        GenerateInput(question="Question 1", ideal_answer="Answer 1"),
        GenerateInput(question="Question 2", ideal_answer="Answer 2"),
    ]
    structured_context = StructuredContext(
        title="Title",
        heading_hierarchy=["Heading 1", "Heading 2"],
        html_content="<p>Some content</p>",
        exact_path="/income-tax",
        base_path="/income-tax",
    )

    expected_results = [
        EvaluationTestCase(
            question="Question 1",
            ideal_answer="Answer 1",
            llm_answer="An answer",
            structured_contexts=[structured_context],
        ),
        EvaluationTestCase(
            question="Question 2",
            ideal_answer="Answer 2",
            llm_answer="An answer",
            structured_contexts=[structured_context],
        ),
    ]
    actual_results = generate_inputs_to_evaluation_test_cases("openai", generate_inputs)

    assert sorted(expected_results, key=lambda r: r.question) == sorted(
        actual_results, key=lambda r: r.question
    )


def test_generate_models_to_evaluation_test_cases_runs_expected_rake_task(
    run_rake_task_mock,
):
    run_rake_task_mock.side_effect = lambda *_: {"message": "An answer", "sources": []}
    generate_inputs = [
        GenerateInput(question="Question 1", ideal_answer="Answer"),
    ]
    generate_inputs_to_evaluation_test_cases("openai", generate_inputs)

    run_rake_task_mock.assert_called_with(
        "evaluation:generate_rag_structured_answer_response[openai]",
        {"INPUT": "Question 1"},
    )


@pytest.mark.usefixtures("run_rake_task_mock")
def test_generate_and_write_dataset(mock_input_data, mock_project_root):
    path = generate_and_write_dataset(mock_input_data, "openai", mock_project_root)
    assert path.exists()
    with open(path, "r") as file:
        for line in file:
            assert json.loads(line)
