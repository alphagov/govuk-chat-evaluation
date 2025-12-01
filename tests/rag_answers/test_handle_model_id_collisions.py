import pytest
from govuk_chat_evaluation.rag_answers.data_models import GenerateInput
from govuk_chat_evaluation.rag_answers.handle_model_id_collisions import (
    ensure_unique_model_ids,
)


def test_inputs_with_unique_ids():
    generate_inputs = [
        GenerateInput(id="question-1", question="Question 1", ideal_answer="Answer 1"),
        GenerateInput(id="question-2", question="Question 2", ideal_answer="Answer 2"),
    ]

    result = ensure_unique_model_ids(generate_inputs)
    assert result is None


def test_inputs_with_colliding_ids():
    generate_inputs = [
        GenerateInput(id="question-1", question="Question 1", ideal_answer="Answer 1"),
        GenerateInput(id="question-1", question="Question 2", ideal_answer="Answer 2"),
        GenerateInput(id="question-2", question="Question 3", ideal_answer="Answer 3"),
        GenerateInput(id="question-2", question="Question 4", ideal_answer="Answer 4"),
    ]

    with pytest.raises(
        ValueError, match="Duplicate IDs found in inputs: question-1, question-2"
    ):
        ensure_unique_model_ids(generate_inputs)
