from pydantic import BaseModel, model_validator
from enum import Enum
from typing import Any
import os

from ...config import BaseConfig
from ...file_system import project_root
from ...aws_credentials import check_aws_credentials
from deepeval.metrics import (
    FaithfulnessMetric,
    BiasMetric,
    BaseMetric,
)
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.models.llms.openai_model import GPTModel
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel

from ..invalid_json_retry import attach_invalid_json_retry_to_model
from ..custom_deepeval.metrics import (
    FactualPrecisionRecall,
    FactualPrecisionRecallMode,
    AbsenceOfFactualContradictions,
    ContextRelevancyMetric,
    CoherenceMetric,
    FactClassificationCache,
)


class MetricName(str, Enum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    BIAS = "bias"
    FACTUAL_PRECISION = "factual_precision"
    FACTUAL_RECALL = "factual_recall"
    ABSENCE_OF_FACTUAL_CONTRADICTIONS = "absence_of_factual_contradictions"
    CONTEXT_RELEVANCY = "context_relevancy"
    COHERENCE = "coherence"
    # others to add


class LLMJudgeModel(str, Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    AMAZON_NOVA_MICRO_1 = "eu.amazon.nova-micro-v1:0"
    AMAZON_NOVA_PRO_1 = "eu.amazon.nova-pro-v1:0"
    GEMINI_15_PRO = "gemini-1.5-pro-002"
    GEMINI_15_FLASH = "gemini-1.5-flash-002"
    GPT_OSS_20B = "openai.gpt-oss-20b-1:0"
    GPT_OSS_120B = "openai.gpt-oss-120b-1:0"


class BedrockCredentialsError(RuntimeError):
    pass


def _ensure_bedrock_credentials(*, region: str) -> None:
    result = check_aws_credentials(region=region)
    if result.ok:
        return

    script_command = "./scripts/export_aws_credentials.sh"
    env_path = project_root() / ".env.aws"
    message_lines = [
        "Missing or invalid AWS credentials for AWS Bedrock judge model.",
        f"Region: {region}",
        f"Run: {script_command} (optionally with a role) or set AWS_* environment variables.",
        f"Expected credentials file at: {env_path}",
    ]
    if result.error:
        message_lines.append(f"Credential check error: {result.error}")

    raise BedrockCredentialsError("\n".join(message_lines))


class LLMJudgeModelConfig(BaseModel):
    model: LLMJudgeModel
    temperature: float = 0.0

    def instantiate_llm_judge(self):
        """Return the LLM judge model instance."""
        match self.model:
            case (
                LLMJudgeModel.AMAZON_NOVA_MICRO_1
                | LLMJudgeModel.AMAZON_NOVA_PRO_1
                | LLMJudgeModel.GPT_OSS_20B
                | LLMJudgeModel.GPT_OSS_120B
            ):
                region = os.getenv("AWS_BEDROCK_REGION", "eu-west-1")
                _ensure_bedrock_credentials(
                    region=region,
                )
                model = AmazonBedrockModel(
                    model=self.model.value,
                    region=region,
                    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                    generation_kwargs={
                        "temperature": self.temperature,
                        "maxTokens": 6000,
                    },
                )
                return attach_invalid_json_retry_to_model(model)
            case LLMJudgeModel.GEMINI_15_PRO:
                raise NotImplementedError(
                    f"Judge model {self.model} instantiation not implemented."
                )
            case LLMJudgeModel.GEMINI_15_FLASH:
                raise NotImplementedError(
                    f"Judge model {self.model} instantiation not implemented."
                )
            case LLMJudgeModel.GPT_4O_MINI | LLMJudgeModel.GPT_4O:
                return GPTModel(model=self.model.value, temperature=self.temperature)


class MetricConfig(BaseModel):
    name: MetricName
    threshold: float
    llm_judge: LLMJudgeModelConfig

    @model_validator(mode="before")
    @classmethod
    def inject_llm_judge(cls, values: dict[str, Any]) -> dict[str, Any]:
        # extract model and temperature to build llm_judge
        if "llm_judge" not in values:
            values["llm_judge"] = {
                "model": values.pop("model"),
                "temperature": values.pop("temperature", 0.0),
            }
        return values


class TaskConfig(BaseConfig):
    what: BaseConfig.GenericFields.what
    generate: BaseConfig.GenericFields.generate
    provider: BaseConfig.GenericFields.provider_openai_or_claude
    input_path: BaseConfig.GenericFields.input_path
    claude_generation_model: BaseConfig.GenericFields.claude_generation_model
    metrics: list[MetricConfig]
    n_runs: int

    @model_validator(mode="after")
    def run_validatons(self):
        return self._validate_fields_required_for_generate("provider")

    def metric_instances(self) -> list[BaseMetric]:
        """Return the list of runtime metric objects for evaluation."""
        fact_classification_cache = FactClassificationCache()
        return [
            self._build_metric(metric, fact_classification_cache)
            for metric in self.metrics
        ]

    def _build_metric(
        self, metric: MetricConfig, fact_classification_cache: FactClassificationCache
    ):
        model = metric.llm_judge.instantiate_llm_judge()
        match metric.name:
            case MetricName.FAITHFULNESS:
                return FaithfulnessMetric(threshold=metric.threshold, model=model)
            case MetricName.RELEVANCE:
                return AnswerRelevancyMetric(threshold=metric.threshold, model=model)
            case MetricName.BIAS:
                return BiasMetric(threshold=metric.threshold, model=model)
            case MetricName.FACTUAL_PRECISION:
                return FactualPrecisionRecall(
                    threshold=metric.threshold,
                    model=model,
                    mode=FactualPrecisionRecallMode.PRECISION,
                    cache=fact_classification_cache,
                )
            case MetricName.FACTUAL_RECALL:
                return FactualPrecisionRecall(
                    threshold=metric.threshold,
                    model=model,
                    mode=FactualPrecisionRecallMode.RECALL,
                    cache=fact_classification_cache,
                )
            case MetricName.ABSENCE_OF_FACTUAL_CONTRADICTIONS:
                return AbsenceOfFactualContradictions(
                    threshold=metric.threshold, model=model
                )
            case MetricName.CONTEXT_RELEVANCY:
                return ContextRelevancyMetric(threshold=metric.threshold, model=model)
            case MetricName.COHERENCE:
                return CoherenceMetric(threshold=metric.threshold, model=model)
