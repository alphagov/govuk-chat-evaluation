from typing import List, Type, TypeVar
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
from .template import ContextRelevancyTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from .schema import (
    TruthCollection,
    InformationNeedsCollection,
    VerdictCollection,
    ScoreReason,
)

SchemaType = TypeVar("SchemaType", bound=BaseModel)


class ContextRelevancyMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: float = 0.8,
        model: str | DeepEvalBaseLLM | None = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[ContextRelevancyTemplate] = ContextRelevancyTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(test_case, self._required_params, self)

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            retrieval_context = test_case.retrieval_context or []
            self.evaluation_cost = 0.0
            truths: TruthCollection = await self._generate_truths(retrieval_context)
            information_needs: InformationNeedsCollection = (
                await self._generate_information_needs(test_case.input)
            )
            verdicts: VerdictCollection = await self._generate_verdicts(
                information_needs, truths
            )
            self.score = self._calculate_score(verdicts)
            self.reason: str | None = await self._generate_reason(
                test_case.input, verdicts
            )
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Truths: {prettify_list(truths.truths)}",
                    f"Information Needs:\n{prettify_list(information_needs.information_needs)}",
                    f"Verdicts:\n{prettify_list(verdicts.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _generate_truths(self, retrieval_context: list[str]) -> TruthCollection:
        prompt = self.evaluation_template.truths(
            retrieval_context=retrieval_context,
        )
        return await self._generate_result_from_model(prompt, schema=TruthCollection)

    async def _generate_information_needs(
        self, input: str
    ) -> InformationNeedsCollection:
        prompt = self.evaluation_template.information_needs(input=input)
        return await self._generate_result_from_model(
            prompt, schema=InformationNeedsCollection
        )

    async def _generate_verdicts(
        self, information_needs: InformationNeedsCollection, truths: TruthCollection
    ) -> VerdictCollection:
        if len(information_needs.information_needs) == 0:
            return VerdictCollection(verdicts=[])

        facts: list[str] = [fact for truth in truths.truths for fact in truth.facts]
        prompt = self.evaluation_template.verdicts(
            information_needs=information_needs.information_needs,
            extracted_truths=facts,
        )

        return await self._generate_result_from_model(prompt, schema=VerdictCollection)

    def _calculate_score(self, verdicts: VerdictCollection) -> float:
        number_of_verdicts = len(verdicts.verdicts)
        if number_of_verdicts == 0:
            return 1

        quality_count = 0
        for verdict in verdicts.verdicts:
            if verdict.verdict != "no":
                quality_count += 1

        score = quality_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    async def _generate_reason(
        self, input: str, verdicts: VerdictCollection
    ) -> str | None:
        if self.include_reason is False:
            return None

        unmet_needs = []
        for verdict in verdicts.verdicts:
            if verdict.verdict == "no":
                unmet_needs.append(verdict.reason)

        score = float(round(self.score or 0.0, 2))
        prompt = self.evaluation_template.reason(
            unmet_needs=unmet_needs,
            input=input,
            score=score,
        )

        result = await self._generate_result_from_model(prompt, schema=ScoreReason)
        return result.reason if isinstance(result.reason, str) else str(result.reason)

    async def _generate_result_from_model(
        self, prompt: str, schema: Type[SchemaType]
    ) -> SchemaType:
        if self.using_native_model:
            result, cost = await self.model.a_generate(prompt, schema=schema)
            self._increment_cost(cost)
            if isinstance(result, schema):
                return result
            else:
                raise TypeError("Generated result does not match the expected schema.")
        else:
            try:
                result = await self.model.a_generate(prompt, schema=schema)
                if isinstance(result, schema):
                    return result
                else:
                    raise TypeError(
                        "Generated result does not match the expected schema."
                    )
            except TypeError:
                result = await self.model.a_generate(prompt)
                data = trimAndLoadJson(result, self)
                model = schema(**data)
                return model

    def is_successful(self) -> bool:
        if self.score is None:
            return False
        else:
            return self.score >= self.threshold

    def _increment_cost(self, cost) -> None:
        if isinstance(cost, float) and isinstance(self.evaluation_cost, float):
            self.evaluation_cost += cost

    @property
    def __name__(self):  # type: ignore[arg-type]
        return "ContextRelevancy"
