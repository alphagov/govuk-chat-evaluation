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
from .template import ContextRelevancyTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from .schema import (
    TruthCollection,
    InformationNeedsCollection,
    VerdictCollection,
    ScoreReason,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    StructuredContext,
)
from deepeval.errors import MissingTestCaseParamsError

SchemaType = TypeVar("SchemaType", bound=BaseModel)


class ContextRelevancyMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
    evaluation_template: Type[ContextRelevancyTemplate] = ContextRelevancyTemplate

    def __init__(
        self,
        threshold: float = 0.8,
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
        **kwargs,  # DeepEval may introduce new kwargs that we don't use
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

        if self.using_native_model:
            self.evaluation_cost = 0.0

        if (
            test_case.additional_metadata is None
            or test_case.additional_metadata.get("structured_contexts") is None
        ):
            raise MissingTestCaseParamsError(
                "additional_metadata['structured_contexts']"
                " cannot be None for ContextRelevancyMetric."
            )

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            structured_contexts = cast(
                List[StructuredContext],
                test_case.additional_metadata["structured_contexts"],
            )

            truth_collection = await self._generate_truths(structured_contexts)
            information_needs_collection = await self._generate_information_needs(
                test_case.input
            )
            verdict_collection = await self._generate_verdicts(
                information_needs_collection, truth_collection
            )
            self.score = self._calculate_score(verdict_collection)
            self.reason = await self._generate_reason(
                test_case.input, verdict_collection
            )
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Truths: {prettify_list(truth_collection.truths)}",
                    f"Information Needs:\n{prettify_list(information_needs_collection.information_needs)}",
                    f"Verdicts:\n{prettify_list(verdict_collection.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _generate_truths(
        self, structured_contexts: list[StructuredContext]
    ) -> TruthCollection:
        retrieval_context = [
            ctx.to_flattened_context_content() for ctx in structured_contexts
        ]
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
        self,
        information_needs_collection: InformationNeedsCollection,
        truth_collection: TruthCollection,
    ) -> VerdictCollection:
        if len(information_needs_collection.information_needs) == 0:
            return VerdictCollection(verdicts=[])

        extracted_truths = [truth.model_dump() for truth in truth_collection.truths]
        prompt = self.evaluation_template.verdicts(
            information_needs=information_needs_collection.information_needs,
            extracted_truths=extracted_truths,
        )

        return await self._generate_result_from_model(prompt, schema=VerdictCollection)

    def _calculate_score(self, verdicts: VerdictCollection) -> float:
        score = verdicts.score_verdicts()

        return 0 if self.strict_mode and score < self.threshold else score

    async def _generate_reason(
        self, input: str, verdict_collection: VerdictCollection
    ) -> str | None:
        if self.include_reason is False:
            return None

        unmet_needs = verdict_collection.unmet_needs()
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
                return cast(SchemaType, result)
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

    @property
    def __name__(self):  # type: ignore[arg-type]
        return "Context Relevancy"
