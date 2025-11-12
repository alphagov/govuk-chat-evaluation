from pydantic import BaseModel, model_validator
from enum import Enum
from typing import Any
import os
from ...config import BaseConfig
from deepeval.metrics import (
    FaithfulnessMetric,
    BiasMetric,
)
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.models.llms.openai_model import GPTModel
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel

from ..invalid_json_retry import attach_invalid_json_retry_to_model
from ..custom_deepeval.metrics import (
    FactualCorrectnessCompleteness,
    FactualMode,
    ContextRelevancyMetric,
    CoherenceMetric,
)


class MetricName(str, Enum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    BIAS = "bias"
    FACTUAL_CORRECTNESS = "factual_correctness"
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


class LLMJudgeModelConfig(BaseModel):
    model: LLMJudgeModel
    temperature: float = 0.0

    def instantiate_llm_judge(self):
        """Return the LLM judge model instance."""
        match self.model:
            case LLMJudgeModel.AMAZON_NOVA_MICRO_1:
                region = os.getenv("AWS_BEDROCK_REGION", "eu-west-1")
                model = AmazonBedrockModel(
                    model_id=self.model.value,
                    region_name=region,
                    generation_kwargs={
                        "temperature": self.temperature,
                        "maxTokens": 6000,
                    },
                )
                return attach_invalid_json_retry_to_model(model)
            case LLMJudgeModel.AMAZON_NOVA_PRO_1:
                region = os.getenv("AWS_BEDROCK_REGION", "eu-west-1")
                model = AmazonBedrockModel(
                    model_id=self.model.value,
                    region_name=region,
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

    def to_metric_instance(self):
        model = self.llm_judge.instantiate_llm_judge()
        match self.name:
            case MetricName.FAITHFULNESS:
                return FaithfulnessMetric(threshold=self.threshold, model=model)
            case MetricName.RELEVANCE:
                return AnswerRelevancyMetric(threshold=self.threshold, model=model)
            case MetricName.BIAS:
                return BiasMetric(threshold=self.threshold, model=model)
            case MetricName.FACTUAL_CORRECTNESS:
                return FactualCorrectnessCompleteness(
                    threshold=self.threshold, model=model, mode=FactualMode.CORRECTNESS
                )
            case MetricName.CONTEXT_RELEVANCY:
                return ContextRelevancyMetric(threshold=self.threshold, model=model)
            case MetricName.COHERENCE:
                return CoherenceMetric(threshold=self.threshold, model=model)


class TaskConfig(BaseConfig):
    what: BaseConfig.GenericFields.what
    generate: BaseConfig.GenericFields.generate
    provider: BaseConfig.GenericFields.provider_openai_or_claude
    input_path: BaseConfig.GenericFields.input_path
    metrics: list[MetricConfig]
    n_runs: int

    @model_validator(mode="after")
    def run_validatons(self):
        return self._validate_fields_required_for_generate("provider")

    def metric_instances(self):
        """Return the list of runtime metric objects for evaluation."""
        return [metric.to_metric_instance() for metric in self.metrics]  # type: ignore
