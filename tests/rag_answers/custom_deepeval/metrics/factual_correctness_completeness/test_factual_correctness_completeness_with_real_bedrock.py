import pytest
from deepeval.test_case import LLMTestCase

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_precision_recall import (
    FactualPrecisionRecall,
    Mode,
)


@pytest.mark.real_bedrock
class TestFactualPrecisionRecallBedrockOpenAI:
    """
    Test the FactualPrecisionRecall with real Bedrock API calls.
    This test requires the Bedrock environment variable to be set with
    `./scripts/export_aws_credentials.sh`.

    It can be run with the command:
    uv run pytest -m 'real_bedrock'
    """

    @pytest.mark.parametrize(
        "mode, llm_test_case, expected_score",
        [
            (
                Mode.PRECISION,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark. Cats Meow.",
                    actual_output="Pigs oink and dogs bark.",
                    input="What noise do pigs and dogs do?",
                ),
                1.0,
            ),
            (
                Mode.PRECISION,
                LLMTestCase(
                    expected_output="Dogs bark.",
                    actual_output="Dogs bark and cats meow.",
                    input="What noise do pigs and dogs do?",
                ),
                0.5,
            ),
            (
                Mode.PRECISION,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs don't bark.",
                    input="What noise do pigs and dogs do?",
                ),
                0.0,
            ),
            (
                Mode.RECALL,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Pigs oink, cats meow and dogs bark.",
                    input="What noise do pigs and dogs do?",
                ),
                1.0,
            ),
            (
                Mode.RECALL,
                LLMTestCase(
                    expected_output="Pigs oink. Dogs bark.",
                    actual_output="Dogs bark.",
                    input="What noise do pigs and dogs do?",
                ),
                0.5,
            ),
            (
                Mode.RECALL,
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
    async def test_factual_precision_recall_score_bedrock(
        self, bedrock_openai_judge, mode: Mode, llm_test_case: LLMTestCase, expected_score: float
    ):
        metric = FactualPrecisionRecall(
            model=bedrock_openai_judge,
            mode=mode,
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)  # type: ignore
        assert round(computed_score, 4) == round(expected_score, 4)
