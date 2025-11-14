import pytest
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_correctness_completeness import (
    FactualCorrectnessCompleteness,
    Mode,
)


@pytest.mark.real_openai
class TestFactualCorrectnessCompletenessRealOpenAI:
    """
    Test the FactualCorrectnessCompleteness with real OpenAI API calls.
    This test requires the OPENAI_API_KEY environment variable to be set.

    It can be run with the command:
    uv run pytest -m 'real_openai'
    """

    @pytest.mark.parametrize(
        "mode, llm_test_case, expected_score",
        [
            (
                Mode.CORRECTNESS,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark. Cats Meow.",
                    actual_output="Pigs oink and dogs bark.",
                    input="What noise do pigs and dogs do?",
                ),
                1.0,
            ),
            (
                Mode.CORRECTNESS,
                LLMTestCase(
                    expected_output="Dogs bark.",
                    actual_output="Dogs bark and cats meow.",
                    input="What noise do pigs and dogs do?",
                ),
                0.5,
            ),
            (
                Mode.CORRECTNESS,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs don't bark.",
                    input="What noise do pigs and dogs do?",
                ),
                0.0,
            ),
            (
                Mode.COMPLETENESS,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Pigs oink, cats meow and dogs bark.",
                    input="What noise do pigs and dogs do?",
                ),
                1.0,
            ),
            (
                Mode.COMPLETENESS,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs bark.",
                    input="What noise do pigs and dogs do?",
                ),
                0.5,
            ),
            (
                Mode.COMPLETENESS,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs don't bark.",
                    input="What noise do pigs and dogs do?",
                ),
                0.0,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_factual_correctness_completeness_score(
        self, mode: Mode, llm_test_case: LLMTestCase, expected_score: float
    ):
        metric = FactualCorrectnessCompleteness(
            model=GPTModel(model="gpt-4o", temperature=0),
            mode=mode,
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)  # type: ignore
        assert round(computed_score, 4) == round(expected_score, 4)
