import pytest
from deepeval.test_case import LLMTestCase

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_precision_recall import (
    FactualPrecisionRecall,
    Mode,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    LLMJudgeModel,
    LLMJudgeModelConfig,
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


LLM_TEST_CASES = [
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
]


class TestFactualPrecisionRecallRealProviders:
    """
    Exercises the FactualPrecisionRecall metric against real providers (OpenAI
    or Bedrock OpenAI).
    Requires corresponding credentials; selection is via markers:

        uv run pytest -m 'real_openai'
        uv run pytest -m 'real_bedrock'
    """

    @pytest.mark.parametrize("llm_judge_config", PROVIDERS)
    @pytest.mark.parametrize(
        "mode, llm_test_case, expected_score",
        LLM_TEST_CASES,
    )
    @pytest.mark.asyncio
    async def test_factual_precision_recall_score(
        self,
        llm_judge_config: LLMJudgeModelConfig,
        mode: Mode,
        llm_test_case: LLMTestCase,
        expected_score: float,
    ):
        model = llm_judge_config.instantiate_llm_judge()
        metric = FactualPrecisionRecall(
            model=model,
            mode=mode,
            include_reason=False,
        )
        computed_score = await metric.a_measure(llm_test_case)  # type: ignore
        assert round(computed_score, 4) == round(expected_score, 4)
