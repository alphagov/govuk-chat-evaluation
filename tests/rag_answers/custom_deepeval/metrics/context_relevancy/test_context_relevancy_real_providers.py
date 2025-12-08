import pytest
from deepeval.test_case import LLMTestCase

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.context_relevancy import (
    ContextRelevancyMetric,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    LLMJudgeModel,
    LLMJudgeModelConfig,
    StructuredContext,
)


PROVIDERS = [
    pytest.param(
        LLMJudgeModelConfig(model=LLMJudgeModel.GPT_4O, temperature=0.0),
        id="openai_gpt-4o",
        marks=pytest.mark.real_openai,
    ),
    pytest.param(
        LLMJudgeModelConfig(model=LLMJudgeModel.GPT_OSS_120B, temperature=0.0),
        id="bedrock_gpt-oss-120b",
        marks=pytest.mark.real_bedrock,
    ),
]


STRUCTURED_CONTEXT = StructuredContext(
    title="Uk inflation data",
    heading_hierarchy=["Inflation", "2024"],
    description="Inflation data by year for the UK. Published by Office for National Statistics.",
    html_content="<p>The inflation rate in the UK is 3.4%. This in an official document published on GOV.UK</p>",
    exact_path="https://gov.uk/inflation-data",
    base_path="https://gov.uk/inflation-data",
)


LLM_TEST_CASES = [
    (
        LLMTestCase(
            input="What is the UK's inflation rate?",
            actual_output="The inflation rate in the UK is 3.4%.",
            additional_metadata={"structured_contexts": [STRUCTURED_CONTEXT]},
        ),
        1.0,
    ),
    (
        LLMTestCase(
            input="Tell me about France.",
            actual_output="It's a country in Europe.",
            additional_metadata={"structured_contexts": [STRUCTURED_CONTEXT]},
        ),
        0.0,
    ),
]


class TestContextRelevancyRealProviders:
    """
    Exercises the ContextRelevancyMetric against real providers (OpenAI or Bedrock OpenAI).
    Requires corresponding credentials; selection is via markers:

        uv run pytest -m 'real_openai'
        uv run pytest -m 'real_bedrock'
    """

    @pytest.mark.parametrize("llm_judge_config", PROVIDERS)
    @pytest.mark.parametrize(
        "llm_test_case, expected_score",
        LLM_TEST_CASES,
    )
    @pytest.mark.asyncio
    async def test_context_relevancy_score(
        self,
        llm_judge_config: LLMJudgeModelConfig,
        llm_test_case: LLMTestCase,
        expected_score: float,
    ):
        model = llm_judge_config.instantiate_llm_judge()
        metric = ContextRelevancyMetric(
            model=model,
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)
        assert round(computed_score, 2) == pytest.approx(expected_score, rel=0.2)
