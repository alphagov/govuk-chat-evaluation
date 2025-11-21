from govuk_chat_evaluation.rag_answers.data_models import GenerateInput
from govuk_chat_evaluation.rag_answers.handle_model_id_collisions import (
    ensure_model_ids_are_unique,
)


def test_inputs_with_unique_ids():
    generate_inputs = [
        GenerateInput(id="question-1", question="Question 1", ideal_answer="Answer 1"),
        GenerateInput(id="question-1", question="Question 2", ideal_answer="Answer 2"),
        GenerateInput(id="question-1", question="Question 3", ideal_answer="Answer 3"),
    ]

    expected_results = [
        GenerateInput(
            id="question-1",
            question="Question 1",
            ideal_answer="Answer 1",
        ),
        GenerateInput(
            id="question-1-duplicate-1",
            question="Question 2",
            ideal_answer="Answer 2",
        ),
        GenerateInput(
            id="question-1-duplicate-2",
            question="Question 3",
            ideal_answer="Answer 3",
        ),
    ]

    actual_results = ensure_model_ids_are_unique(generate_inputs)
    assert expected_results == actual_results
