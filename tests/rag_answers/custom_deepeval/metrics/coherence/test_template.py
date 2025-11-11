from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.coherence.template import (
    CoherenceTemplate,
    RUBRIC,
    SCORE_RANGE,
)


class TestCoherenceTemplate:
    def test_evaluate_includes_rubric(self):
        user_input = "question"
        actual_output = "response"

        template = CoherenceTemplate.evaluate(user_input, actual_output)

        assert f"between {SCORE_RANGE[0]} and {SCORE_RANGE[1]}" in template
        for i in range(len(RUBRIC)):
            assert f"{i + 1}: {RUBRIC[i]}" in template

    def test_evaluate_includes_input_and_output(self):
        user_input = "What is Universal Credit?"
        actual_output = "Universal Credit is a payment to help with your living costs if you're on a low income or out of work."

        template = CoherenceTemplate.evaluate(user_input, actual_output)

        assert user_input in template
        assert actual_output in template
