import pytest
from deepeval.test_case import LLMTestCase

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.absence_of_factual_contradictions.absence_of_factual_contradictions import (
    AbsenceOfFactualContradictions,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    LLMJudgeModel,
    LLMJudgeModelConfig,
)


PROVIDERS = [
    pytest.param(
        LLMJudgeModelConfig(model=LLMJudgeModel.GPT_OSS_120B, temperature=0.0),
        id="bedrock_gpt-oss-120b",
        marks=pytest.mark.real_bedrock,
    ),
]


LLM_TEST_CASES = [
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
]


class TestAbsenceOfFactualContradictionsRealProviders:
    """
    Exercises the AbsenceOfFactualContradictions metric against a real Bedrock provider.
    Requires AWS credentials; selection is via marker:

        uv run pytest -m 'real_bedrock'
    """

    @pytest.mark.parametrize("llm_judge_config", PROVIDERS)
    @pytest.mark.parametrize(
        "llm_test_case, expected_score",
        LLM_TEST_CASES,
    )
    @pytest.mark.asyncio
    async def test_absence_of_factual_contradictions_score(
        self,
        llm_judge_config: LLMJudgeModelConfig,
        llm_test_case: LLMTestCase,
        expected_score: float,
    ):
        model = llm_judge_config.instantiate_llm_judge()
        metric = AbsenceOfFactualContradictions(
            model=model,
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)
        assert round(computed_score, 2) == pytest.approx(expected_score, rel=0.2)
