import pytest
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.absence_of_factual_contradictions.absence_of_factual_contradictions import (
    AbsenceOfFactualContradictions,
)


@pytest.mark.real_openai
class TestAbsenceOfFactualContradictionsRealOpenAI:
    """
    Test the AbsenceOfFactualContradictions metric with real OpenAI API calls.
    This test requires the OPENAI_API_KEY environment variable to be set.

    Run with:
        uv run pytest -m 'real_openai'
    """

    @pytest.mark.parametrize(
        "llm_test_case, expected_score",
        [
            (
                LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="Paris is the capital of France.",
                    expected_output="The capital of France is Paris.",
                ),
                1.0,
            ),
            (
                LLMTestCase(
                    input="What is the capital of France?",
                    actual_output="London is the capital of France.",
                    expected_output="The capital of France is Paris.",
                ),
                0.0,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_absence_of_factual_contradictions_score(
        self, llm_test_case: LLMTestCase, expected_score: float
    ):
        metric = AbsenceOfFactualContradictions(
            model=GPTModel(model="gpt-4o", temperature=0),
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)
        assert round(computed_score, 2) == pytest.approx(expected_score, rel=0.2)
