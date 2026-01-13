from typing import List, Type, TypeVar, cast, Optional
from pydantic import BaseModel

from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics import BaseMetric
from deepeval.utils import prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from .template import ContradictionsTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from .schema import (
    TruthCollection,
    ClaimCollection,
    VerdictCollection,
    ScoreReason,
)

SchemaType = TypeVar("SchemaType", bound=BaseModel)


class AbsenceOfFactualContradictions(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]
    evaluation_template: Type[ContradictionsTemplate] = ContradictionsTemplate

    def __init__(
        self,
        threshold: float = 0.5,
        model: str | DeepEvalBaseLLM | None = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        **kwargs,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            truth_collection = await self._generate_truths(
                cast(str, test_case.expected_output)
            )
            self.truths = truth_collection
            self.claims = await self._generate_claims(
                cast(str, test_case.actual_output)
            )
            self.verdicts = await self._generate_verdicts(self.claims, truth_collection)

            self.score = self._calculate_score(self.verdicts)
            self.reason = await self._generate_reason(self.score, self.verdicts)
            self.success = self.is_successful()

            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Truths:\n{prettify_list(truth_collection.truths)}",
                    f"Claims:\n{prettify_list(self.claims.claims)}",
                    f"Verdicts:\n{prettify_list(self.verdicts.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _generate_truths(self, expected_output: str) -> TruthCollection:
        prompt = self.evaluation_template.generate_truths(text=expected_output)
        return await self._generate_result_from_model(prompt, schema=TruthCollection)

    async def _generate_claims(self, actual_output: str) -> ClaimCollection:
        prompt = self.evaluation_template.generate_claims(text=actual_output)
        return await self._generate_result_from_model(prompt, schema=ClaimCollection)

    async def _generate_verdicts(
        self, claims: ClaimCollection, truths: TruthCollection
    ) -> VerdictCollection:
        if len(claims.claims) == 0:
            return VerdictCollection(verdicts=[])

        prompt = self.evaluation_template.generate_verdicts(
            claims=claims.claims, ground_truth=truths.truths
        )
        return await self._generate_result_from_model(prompt, schema=VerdictCollection)

    async def _generate_reason(
        self, score: float, verdicts: VerdictCollection
    ) -> str | None:
        if self.include_reason is False:
            return None

        contradictions = verdicts.contradiction_reasons()

        prompt = self.evaluation_template.generate_reason(
            score=float(format(score, ".2f")),
            contradictions=contradictions,
        )

        result = await self._generate_result_from_model(prompt, schema=ScoreReason)
        return result.reason

    def _calculate_score(self, verdicts: VerdictCollection) -> float:
        score = verdicts.score_verdicts()
        return 0 if self.strict_mode and score < self.threshold else score

    async def _generate_result_from_model(
        self, prompt: str, schema: Type[SchemaType]
    ) -> SchemaType:
        if self.using_native_model:
            result, cost = cast(
                tuple[SchemaType, Optional[float]],
                await self.model.a_generate(prompt, schema=schema),
            )
            if isinstance(cost, float):
                self.evaluation_cost = (self.evaluation_cost or 0.0) + cost
            return result
        else:
            try:
                return cast(
                    SchemaType,
                    await self.model.a_generate(prompt, schema=schema),
                )
            except TypeError:
                result = await self.model.a_generate(prompt)
                data = trimAndLoadJson(result, self)
                model = schema(**data)
                return model

    def is_successful(self) -> bool:
        if self.score is None:
            return False
        return self.score >= self.threshold

    @property
    def __name__(self):  # type: ignore
        return "Absence of Factual Contradictions"
