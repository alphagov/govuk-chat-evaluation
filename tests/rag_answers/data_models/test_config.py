import pytest
from pydantic import ValidationError
from deepeval.metrics import (
    FaithfulnessMetric,
    BiasMetric,
)
from deepeval.models.llms.openai_model import GPTModel
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel

from govuk_chat_evaluation.rag_answers.data_models import (
    config as config_module,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    MetricName,
    LLMJudgeModel,
    LLMJudgeModelConfig,
    MetricConfig,
    TaskConfig,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.coherence import (
    CoherenceMetric,
)
from govuk_chat_evaluation.rag_answers.patch_bedrock_a_generate import (
    a_generate_filters_non_text_responses,
)


def test_config_monkeypatches_amazong_bedrock_model_a_generate_on_initialisation():
    assert AmazonBedrockModel.a_generate == a_generate_filters_non_text_responses


class TestMetricConfig:
    @pytest.mark.parametrize(
        "config_dict, expected_class",
        [
            (
                {
                    "name": "faithfulness",
                    "threshold": 0.8,
                    "model": "gpt-4o",
                    "temperature": 0.0,
                },
                FaithfulnessMetric,
            ),
            (
                {
                    "name": "bias",
                    "threshold": 0.5,
                    "model": "gpt-4o",
                    "temperature": 0.0,
                },
                BiasMetric,
            ),
            (
                {
                    "name": "coherence",
                    "threshold": 0.8,
                    "model": "gpt-4o",
                    "temperature": 0.0,
                },
                CoherenceMetric,
            ),
        ],
    )
    def test_to_metric_instance(self, config_dict, expected_class):
        metric_config = MetricConfig(**config_dict)
        assert isinstance(metric_config.to_metric_instance(), expected_class)

    @pytest.mark.parametrize(
        "judge_model, expected_llm_cls",
        [
            (LLMJudgeModel.GPT_4O, GPTModel),
            (LLMJudgeModel.GPT_4O_MINI, GPTModel),
            (LLMJudgeModel.AMAZON_NOVA_MICRO_1, AmazonBedrockModel),
            (LLMJudgeModel.AMAZON_NOVA_PRO_1, AmazonBedrockModel),
            (LLMJudgeModel.GPT_OSS_20B, AmazonBedrockModel),
            (LLMJudgeModel.GPT_OSS_120B, AmazonBedrockModel),
        ],
    )
    def test_to_metric_instance_instantiates_llm_model(
        self, judge_model, expected_llm_cls
    ):
        metric_config = MetricConfig(
            name=MetricName.FAITHFULNESS,
            threshold=0.5,
            llm_judge=LLMJudgeModelConfig(model=judge_model, temperature=0.0),
        )
        metric = metric_config.to_metric_instance()
        assert isinstance(metric.model, expected_llm_cls)

    @pytest.mark.parametrize(
        "judge_model",
        [
            LLMJudgeModel.AMAZON_NOVA_MICRO_1,
            LLMJudgeModel.AMAZON_NOVA_PRO_1,
            LLMJudgeModel.GPT_OSS_20B,
            LLMJudgeModel.GPT_OSS_120B,
        ],
    )
    def test_to_metric_instance_monkeypatches_bedrock_models(self, mocker, judge_model):
        retry_path = (
            "govuk_chat_evaluation.rag_answers.data_models.config."
            "attach_invalid_json_retry_to_model"
        )
        wrapped_retry = mocker.patch(
            retry_path,
            wraps=config_module.attach_invalid_json_retry_to_model,
        )
        metric_config = MetricConfig(
            name=MetricName.FAITHFULNESS,
            threshold=0.5,
            llm_judge=LLMJudgeModelConfig(model=judge_model, temperature=0.0),
        )

        metric = metric_config.to_metric_instance()

        wrapped_retry.assert_called_once_with(metric.model)

    def test_get_metric_instance_invalid_enum(self):
        config_dict = {
            "name": "does_not_exist",
            "threshold": 0.5,
            "model": "gpt-4o",
            "temperature": 0.0,
        }

        with pytest.raises(ValidationError) as exception_info:
            MetricConfig(**config_dict)

        assert "validation error for MetricConfig" in str(exception_info.value)
        assert "does_not_exist" in str(exception_info.value)


class TestTaskConfig:
    def test_config_requires_provider_for_generate(self, mock_input_data):
        with pytest.raises(ValueError, match="provider is required to generate data"):
            TaskConfig(
                what="Test",
                generate=True,
                provider=None,
                input_path=mock_input_data,
                metrics=[],
                n_runs=1,
            )

        # These should not raise
        TaskConfig(
            what="Test",
            generate=False,
            provider=None,
            input_path=mock_input_data,
            metrics=[],
            n_runs=1,
        )

        TaskConfig(
            what="Test",
            generate=True,
            provider="openai",
            input_path=mock_input_data,
            metrics=[],
            n_runs=1,
        )

    def test_get_metric_instances(self, mock_input_data):
        config_dict = {
            "what": "Test",
            "generate": False,
            "provider": None,
            "input_path": mock_input_data,
            "metrics": [
                {
                    "name": "faithfulness",
                    "threshold": 0.8,
                    "model": "gpt-4o",
                    "temperature": 0.0,
                },
                {
                    "name": "bias",
                    "threshold": 0.5,
                    "model": "gpt-4o",
                    "temperature": 0.5,
                },
            ],
            "n_runs": 3,
        }

        evaluation_config = TaskConfig(**config_dict)
        metrics = evaluation_config.metric_instances()
        assert len(metrics) == 2
        assert isinstance(metrics[0], FaithfulnessMetric)
        assert isinstance(metrics[1], BiasMetric)
