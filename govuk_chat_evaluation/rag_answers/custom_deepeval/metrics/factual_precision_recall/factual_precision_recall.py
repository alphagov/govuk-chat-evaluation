from typing import Optional, List, Type
from enum import Enum, auto

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric

from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.telemetry import capture_metric_type

from .template import (
    FactualPrecisionRecallTemplate,
)
from .schema import ClassifiedFacts, FactClassificationResult
from .cache import FactClassificationCache
import logging


class Mode(Enum):
    PRECISION = auto()
    RECALL = auto()


class FactualPrecisionRecall(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    evaluation_template: Type[FactualPrecisionRecallTemplate] = (
        FactualPrecisionRecallTemplate
    )
    async_mode: bool = True

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        mode: Mode,
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        cache: FactClassificationCache | None = None,
        verbose_mode: bool = False,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.mode = mode
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_cost: float | None = None
        self.confusion_matrix: ClassifiedFacts = ClassifiedFacts()
        self.cache = cache or FactClassificationCache()
        self.verbose_mode = verbose_mode
        self._used_cache_for_last_classification = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Synchronously evaluate the metric (precision or recall) for a test case."""
        raise NotImplementedError(
            "Synchronous evaluation is not supported. Use async a_measure instead."
        )

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        **kwargs,  # DeepEval may introduce new kwargs that we don't use
    ) -> float:
        """Asynchronously evaluate factual precision or recall, depending on `mode`."""
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

        if self.using_native_model:
            self.evaluation_cost = 0.0

        with metric_progress_indicator(
            self,
            async_mode=self.async_mode,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.confusion_matrix = await self._a_classify_statements(
                test_case.input,
                test_case.actual_output or "",
                test_case.expected_output or "",
            )
            logging.debug(
                f"Confusion matrix for test input: '{test_case.input}': \n{self.confusion_matrix}"
            )
            return self._finalise_evaluation(test_case.input)

    def _finalise_evaluation(self, input: str) -> float:
        """Finalise the evaluation by computing score, reason, and success status."""
        if self.confusion_matrix.has_facts():
            self.score = self._calculate_score()
            self.reason = self._generate_reason()
            self.success = self.score >= self.threshold
            capture_metric_type(
                self.__name__, async_mode=self.async_mode, in_component=False
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Mode: {self.__name__}",
                    f"Cache hit: {self._used_cache_for_last_classification}",
                    (
                        "TP: "
                        f"{len(self.confusion_matrix.TP)}, "
                        f"FP: {len(self.confusion_matrix.FP)}, "
                        f"FN: {len(self.confusion_matrix.FN)}"
                    ),
                    f"Score: {self.score}",
                    f"Reason: {self.reason or 'Reason omitted'}",
                ],
            )
            return self.score
        else:
            self.error = f"Error: no facts were classified. confusion_matrix is empty for input: {input}."
            logging.error(self.error)
            return float("nan")

    def _generate_reason(self) -> Optional[str]:
        if not self.include_reason or not self.confusion_matrix.has_facts():
            return None
        return f'{{"true_positive_statements": {self.confusion_matrix.TP}, "false_positive_statements": {self.confusion_matrix.FP}}}'

    async def _a_classify_statements(
        self, input: str, actual_output: str, expected_output: str
    ) -> ClassifiedFacts:
        cached = self.cache.get(self.evaluation_model, actual_output, expected_output)
        if cached is not None:
            self._used_cache_for_last_classification = True
            return cached

        self._used_cache_for_last_classification = False

        prompt = self.evaluation_template.classify_facts(
            answer=actual_output, ground_truth=expected_output
        )
        classified_facts: ClassifiedFacts
        should_cache_result = True
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=FactClassificationResult
            )
            if isinstance(cost, float):
                self.evaluation_cost = (self.evaluation_cost or 0.0) + cost
            classified_facts = res.classified_facts  # type: ignore[arg-type]
        else:
            try:
                res = await self.model.a_generate(
                    prompt, schema=FactClassificationResult
                )
                classified_facts = res.classified_facts  # type: ignore[arg-type]
            except TypeError:
                try:
                    res = await self.model.a_generate(prompt)
                    data = trimAndLoadJson(res, self)
                    data_model = FactClassificationResult(**data)
                    classified_facts = data_model.classified_facts
                except Exception as inner_e:
                    logging.error(
                        f"Failed to parse fallback JSON for test input: {input}",
                        exc_info=inner_e,
                    )
                    classified_facts = ClassifiedFacts()
                    should_cache_result = False

        if should_cache_result:
            self.cache.set(
                self.evaluation_model,
                actual_output,
                expected_output,
                classified_facts,
            )
        return classified_facts

    def _calculate_score(self) -> float:
        """
        Calculate the factual-precision or factual-recall score.

        The score is derived from the confusion matrix as a float between 0 and 1.
        Depending on `self.mode`, it is calculated as:
        - Precision: the ratio of the number of True Positive statements to the
          total number of True Positive + False Positive statements.
        - Recall: the ratio of the number of True Positive statements to the
          total number of True Positive + False Negative statements.

        Returns:
            float: The factual-precision or the factual-recall score, depending
            on `mode`.
        """
        if self.confusion_matrix is None:
            return 0.0

        tp = len(self.confusion_matrix.TP)
        fp = len(self.confusion_matrix.FP)
        fn = len(self.confusion_matrix.FN)
        match self.mode:
            case Mode.PRECISION:
                score = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            case Mode.RECALL:
                score = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        return 0.0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return bool(self.success)

    @property
    def __name__(self):  # type: ignore[arg-type]
        match self.mode:
            case Mode.RECALL:
                return "Factual Recall"
            case Mode.PRECISION:
                return "Factual Precision"
