import pytest
from pydantic import ValidationError
from deepeval.metrics import (
    FaithfulnessMetric,
    BiasMetric,
)
from deepeval.models.llms.openai_model import GPTModel
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel

from typing import cast

from govuk_chat_evaluation.rag_answers.data_models import (
    MetricName,
    LLMJudgeModel,
    LLMJudgeModelConfig,
    MetricConfig,
    TaskConfig,
    config as config_module,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics import (
    FactualPrecisionRecall,
    FactClassificationCache,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.coherence import (
    CoherenceMetric,
)
from govuk_chat_evaluation.aws_credentials import AwsCredentialCheckResult


class TestMetricConfig:
    @pytest.fixture
    def task_config(self, mock_input_data):
        return TaskConfig(
            what="Test",
            generate=False,
            provider=None,
            input_path=mock_input_data,
            metrics=[],
            n_runs=1,
        )

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
    def test_build_metric(self, config_dict, expected_class, task_config):
        metric_config = MetricConfig(**config_dict)
        metric = task_config._build_metric(metric_config, FactClassificationCache())
        assert isinstance(metric, expected_class)

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
    def test_build_metric_instantiates_llm_model(
        self, judge_model, expected_llm_cls, task_config, mocker
    ):
        aws_check = mocker.patch(
            "govuk_chat_evaluation.rag_answers.data_models.config.check_aws_credentials",
            return_value=AwsCredentialCheckResult(ok=True),
        )
        metric_config = MetricConfig(
            name=MetricName.FAITHFULNESS,
            threshold=0.5,
            llm_judge=LLMJudgeModelConfig(model=judge_model, temperature=0.0),
        )
        metric = task_config._build_metric(metric_config, FactClassificationCache())
        assert isinstance(metric.model, expected_llm_cls)
        if judge_model in {
            LLMJudgeModel.AMAZON_NOVA_MICRO_1,
            LLMJudgeModel.AMAZON_NOVA_PRO_1,
            LLMJudgeModel.GPT_OSS_20B,
            LLMJudgeModel.GPT_OSS_120B,
        }:
            aws_check.assert_called_once()
        else:
            aws_check.assert_not_called()

    @pytest.mark.parametrize(
        "judge_model",
        [
            LLMJudgeModel.AMAZON_NOVA_MICRO_1,
            LLMJudgeModel.AMAZON_NOVA_PRO_1,
            LLMJudgeModel.GPT_OSS_20B,
            LLMJudgeModel.GPT_OSS_120B,
        ],
    )
    def test_build_metric_monkeypatches_bedrock_models(
        self, mocker, judge_model, task_config
    ):
        aws_check = mocker.patch(
            "govuk_chat_evaluation.rag_answers.data_models.config.check_aws_credentials",
            return_value=AwsCredentialCheckResult(ok=True),
        )
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

        metric = task_config._build_metric(metric_config, FactClassificationCache())

        wrapped_retry.assert_called_once_with(metric.model)
        aws_check.assert_called_once()

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

    def test_fact_classification_cache_scoped_per_metric_instances_call(
        self, mock_input_data
    ):
        config_dict = {
            "what": "Test",
            "generate": False,
            "provider": None,
            "input_path": mock_input_data,
            "metrics": [
                {
                    "name": "factual_precision",
                    "threshold": 0.5,
                    "model": "gpt-4o",
                    "temperature": 0.0,
                },
                {
                    "name": "factual_recall",
                    "threshold": 0.5,
                    "model": "gpt-4o",
                    "temperature": 0.0,
                },
            ],
            "n_runs": 1,
        }

        config = TaskConfig(**config_dict)

        first_run_metrics = config.metric_instances()
        second_run_metrics = config.metric_instances()

        assert isinstance(first_run_metrics[0], FactualPrecisionRecall)
        assert isinstance(first_run_metrics[1], FactualPrecisionRecall)
        assert isinstance(second_run_metrics[0], FactualPrecisionRecall)
        assert isinstance(second_run_metrics[1], FactualPrecisionRecall)

        fc1 = cast(FactualPrecisionRecall, first_run_metrics[0])
        fc2 = cast(FactualPrecisionRecall, first_run_metrics[1])
        fc3 = cast(FactualPrecisionRecall, second_run_metrics[0])
        fc4 = cast(FactualPrecisionRecall, second_run_metrics[1])

        # Within a single call, factual metrics share the same cache
        assert fc1.cache is fc2.cache
        assert fc3.cache is fc4.cache

        # Across calls, a fresh cache is created
        assert fc1.cache is not fc3.cache

    def test_instantiate_llm_judge_raises_on_bedrock_credentials_error(self, mocker):
        mocker.patch(
            "govuk_chat_evaluation.rag_answers.data_models.config.check_aws_credentials",
            return_value=AwsCredentialCheckResult(ok=False, error="ExpiredToken"),
        )

        llm_config = LLMJudgeModelConfig(
            model=LLMJudgeModel.GPT_OSS_120B, temperature=0.0
        )

        with pytest.raises(config_module.BedrockCredentialsError) as exc_info:
            llm_config.instantiate_llm_judge()

        message = str(exc_info.value)
        assert "ExpiredToken" in message
        assert "export_aws_credentials.sh" in message
