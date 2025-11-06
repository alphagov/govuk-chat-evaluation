import pytest
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel
from deepeval import assert_test
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.context_relevancy import (
    ContextRelevancyMetric,
)
from govuk_chat_evaluation.rag_answers.data_models import StructuredContext


@pytest.mark.real_openai
class TestContextRelevancyRealOpenAI:
    """
    Test the ContextRelevancyMetric with real OpenAI API calls.
    This test requires the OPENAI_API_KEY environment variable to be set.

    Run with:
        uv run pytest -m 'real_openai'
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
                    additional_metadata={"structured_contexts": [structured_context]},
                ),
                1.0,
            ),
            (
                LLMTestCase(
                    input="Tell me about France.",
                    actual_output="It's a country in Europe.",
                    additional_metadata={"structured_contexts": [structured_context]},
                ),
                0.0,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_context_relevancy_score(
        self, llm_test_case: LLMTestCase, expected_score: float
    ):
        metric = ContextRelevancyMetric(
            model=GPTModel(model="gpt-4o", temperature=0),
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)
        assert round(computed_score, 2) == pytest.approx(expected_score, rel=0.2)

    def test_context_relevancy_deepeval(self):
        test_case = LLMTestCase(
            input="What is the UK's inflation rate?",
            actual_output="The inflation rate in the UK is 3.4%.",
            additional_metadata={"structured_contexts": [self.structured_context]},
        )

        metric = ContextRelevancyMetric(
            model=GPTModel(model="gpt-4o", temperature=0),
            include_reason=False,
        )

        assert_test(test_case, [metric])
