import pytest
from deepeval.test_case import LLMTestCase

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.context_relevancy import (
    ContextRelevancyMetric,
)
from govuk_chat_evaluation.rag_answers.data_models import StructuredContext


@pytest.mark.real_bedrock
class TestContextRelevancyBedrockOpenAI:
    """
    Test the ContextRelevancyMetric with real Bedrock API calls.
    This test requires the Bedrock environment variable to be set with
    `./scripts/export_aws_credentials.sh`.

    It can be run with the command:
    uv run pytest -m 'real_bedrock'
    """

    structured_context = StructuredContext(
        title="Uk inflation data",
        heading_hierarchy=["Inflation", "2024"],
        description="Inflation data by year for the UK. Published by Office for National Statistics.",
        html_content="<p>The inflation rate in the UK is 3.4%. This in an official document published on GOV.UK</p>",
        exact_path="https://gov.uk/inflation-data",
        base_path="https://gov.uk/inflation-data",
    )

    @pytest.mark.parametrize(
        "llm_test_case, expected_score",
        [
            (
                LLMTestCase(
                    input="What is the UK's inflation rate?",
                    actual_output="The inflation rate in the UK is 3.4%.",
                    additional_metadata={
                        "structured_contexts": [structured_context]
                    },
                ),
                1.0,
            ),
            (
                LLMTestCase(
                    input="Tell me about France.",
                    actual_output="It's a country in Europe.",
                    additional_metadata={
                        "structured_contexts": [structured_context]
                    },
                ),
                0.0,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_context_relevancy_score_bedrock(
        self, bedrock_openai_judge, llm_test_case: LLMTestCase, expected_score: float
    ):
        metric = ContextRelevancyMetric(
            model=bedrock_openai_judge,
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)
        assert round(computed_score, 2) == pytest.approx(expected_score, rel=0.2)
