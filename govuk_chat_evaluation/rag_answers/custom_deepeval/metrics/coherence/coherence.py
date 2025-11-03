from typing import Type, cast

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
    initialize_model,
    trimAndLoadJson,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.telemetry import capture_metric_type
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from .schema import CoherenceJudgement
from .template import CoherenceTemplate, SCORE_RANGE

COHERENCE_THRESHOLD = 0.8


class CoherenceMetric(BaseMetric):
    _required_params: list[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
    evaluation_template: Type[CoherenceTemplate] = CoherenceTemplate

    def __init__(
        self,
        model: DeepEvalBaseLLM | str | None,
        threshold: float = COHERENCE_THRESHOLD,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.rubric_score: int | None = None

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)

        if self.using_native_model:
            self.evaluation_cost = 0.0

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            prompt = self.evaluation_template.evaluate(
                user_input=test_case.input or "",
                actual_output=test_case.actual_output or "",
            )

            judgement = await self._generate_result_from_model(
                prompt, schema=CoherenceJudgement
            )

            self.rubric_score = self._clamp_score(judgement.score)
            self.score = self._normalise_score(self.rubric_score)
            cleaned_reason = judgement.reason.strip()
            self.reason = cleaned_reason if self.include_reason else None
            self.success = self.is_successful()

            capture_metric_type(
                self.__name__, async_mode=True, in_component=_in_component
            )

            verbose_reason = cleaned_reason if self.include_reason else "Reason omitted"
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Rubric Score: {self.rubric_score}",
                    f"Score: {self.score}",
                    f"Reason: {verbose_reason}",
                ],
            )

            return self.score

    async def _generate_result_from_model(
        self, prompt: str, schema: Type[CoherenceJudgement]
    ) -> CoherenceJudgement:
        if self.using_native_model:
            result, cost = cast(
                tuple[CoherenceJudgement, float],
                await self.model.a_generate(prompt, schema=schema),
            )
            self.evaluation_cost = (self.evaluation_cost or 0.0) + cost
            return result

        try:
            return cast(
                CoherenceJudgement,
                await self.model.a_generate(prompt, schema=schema),
            )
        except TypeError:
            raw_response = await self.model.a_generate(prompt)
            data = trimAndLoadJson(raw_response, self)
            return schema(**data)

    def _normalise_score(self, rubric_score: int) -> float:
        """Normalise rubric score (1–5) to 0–1 so thresholds/rollups align with other metrics."""
        lo, hi = SCORE_RANGE
        return (rubric_score - lo) / (hi - lo)

    def is_successful(self) -> bool:
        if self.score is None:
            return False
        return self.score >= self.threshold

    @property
    def __name__(self):  # type: ignore[override]
        return "Coherence"

    def _clamp_score(self, score: int) -> int:
        lo, hi = SCORE_RANGE
        return max(lo, min(hi, score))
