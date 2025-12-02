import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.errors import MissingTestCaseParamsError
from deepeval.utils import prettify_list
import re

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.context_relevancy import (
    ContextRelevancyMetric,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.context_relevancy.schema import (
    Truth,
    TruthCollection,
    InformationNeedsCollection,
    Verdict,
    VerdictCollection,
    ScoreReason,
)
from govuk_chat_evaluation.rag_answers.data_models import (
    StructuredContext,
)


@pytest.fixture
def mock_native_model(
    sample_truths, sample_information_needs, sample_verdicts, sample_reason
):
    mock = Mock(spec=GPTModel)
    mock.get_model_name.return_value = "eu.amazon.nova-pro-v1:0"
    mock.a_generate = AsyncMock()
    mock.a_generate.side_effect = [
        (sample_truths, 0.1),
        (sample_information_needs, 0.1),
        (sample_verdicts, 0.1),
        (sample_reason, 0.1),
    ]
    return mock


@pytest.fixture
def mock_non_native_model(
    sample_truths, sample_information_needs, sample_verdicts, sample_reason
):
    mock = Mock(spec=DeepEvalBaseLLM)
    mock.get_model_name.return_value = "non-native-model"
    mock.a_generate = AsyncMock()
    mock.a_generate.side_effect = [
        (sample_truths, 0.1),
        (sample_information_needs, 0.1),
        (sample_verdicts, 0.1),
        (sample_reason, 0.1),
    ]
    return mock


@pytest.fixture
def mock_test_case():
    structured_context = StructuredContext(
        title="VAT",
        heading_hierarchy=["Tax", "VAT"],
        description="VAT overview",
        html_content="<p>Some HTML about VAT</p>",
        exact_path="https://gov.uk/vat",
        base_path="https://gov.uk",
    )
    return LLMTestCase(
        input="What is the UK's inflation rate?",
        actual_output="The inflation rate is 3.4%.",
        additional_metadata={"structured_contexts": [structured_context]},
    )


@pytest.fixture
def sample_truths():
    return TruthCollection(
        truths=[Truth(context="Context", facts=["UK inflation 3.4%"])]
    )


@pytest.fixture
def sample_information_needs():
    return InformationNeedsCollection(information_needs=["Find UK's inflation rate."])


@pytest.fixture
def sample_verdicts():
    return VerdictCollection(
        verdicts=[
            Verdict(verdict="yes", reason="Matches inflation fact."),
            Verdict(verdict="no", reason="Irrelevant context."),
        ]
    )


@pytest.fixture
def sample_reason():
    return ScoreReason(reason="The answer aligns with most contexts.")


class TestContextRelevancyMetric:
    class TestAMeasure:
        @pytest.mark.asyncio
        async def test_invalid_params_raise_error(self, mock_native_model):
            metric = ContextRelevancyMetric(model=mock_native_model)
            invalid_case = LLMTestCase(input="question", actual_output=None)
            with pytest.raises(MissingTestCaseParamsError, match="cannot be None"):
                await metric.a_measure(invalid_case)

        @pytest.mark.asyncio
        async def test_no_additional_metadata_raises_error(self, mock_native_model):
            metric = ContextRelevancyMetric(model=mock_native_model)
            invalid_case = LLMTestCase(
                input="question", actual_output="answer", additional_metadata=None
            )
            with pytest.raises(
                MissingTestCaseParamsError,
                match=re.escape(
                    "additional_metadata['structured_contexts'] cannot be None for ContextRelevancyMetric."
                ),
            ):
                await metric.a_measure(invalid_case)

        @pytest.mark.asyncio
        async def test_no_structured_context_metadata_raises_error(
            self, mock_native_model
        ):
            metric = ContextRelevancyMetric(model=mock_native_model)
            invalid_case = LLMTestCase(
                input="question", actual_output="answer", additional_metadata={}
            )
            with pytest.raises(
                MissingTestCaseParamsError,
                match=re.escape(
                    "additional_metadata['structured_contexts'] cannot be None for ContextRelevancyMetric."
                ),
            ):
                await metric.a_measure(invalid_case)

        @pytest.mark.asyncio
        @patch(
            "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.context_relevancy.context_relevancy.metric_progress_indicator"
        )
        @pytest.mark.parametrize(
            "set_show_progress, expected_show_progress",
            [
                (True, True),
                (False, False),
            ],
        )
        async def test_show_progress(
            self,
            mock_progress_indicator,
            mock_native_model,
            mock_test_case,
            set_show_progress: bool,
            expected_show_progress: bool,
        ):
            metric = ContextRelevancyMetric(model=mock_native_model)
            # calling the patched metric_progress_indicator creates the mock_progress_indicator
            await metric.a_measure(mock_test_case, _show_indicator=set_show_progress)

            mock_progress_indicator.assert_called_once_with(
                metric,
                async_mode=metric.async_mode,
                _show_indicator=expected_show_progress,
                _in_component=False,
            )

        @pytest.mark.asyncio
        async def test_returns_correct_score(
            self,
            mock_native_model,
            mock_test_case,
        ):
            metric = ContextRelevancyMetric(
                model=mock_native_model, include_reason=True
            )
            score = await metric.a_measure(mock_test_case)

            # 2 verdicts: 1 yes, 1 no => score = 1 / 2 = 0.5
            assert score == 0.5

        @pytest.mark.asyncio
        async def test_tracks_evaluation_cost_for_native_models(
            self,
            mock_native_model,
            mock_test_case,
            sample_truths,
            sample_information_needs,
            sample_verdicts,
            sample_reason,
        ):
            metric = ContextRelevancyMetric(model=mock_native_model)
            await metric.a_measure(mock_test_case)

            assert metric.evaluation_cost == 0.4

            # Second evaluation should reset cost back to 0.0 and recount
            mock_native_model.a_generate.reset_mock()
            mock_native_model.a_generate.side_effect = [
                (sample_truths, 0.05),
                (sample_information_needs, 0.05),
                (sample_verdicts, 0.05),
                (sample_reason, 0.05),
            ]

            await metric.a_measure(mock_test_case)
            assert metric.evaluation_cost == pytest.approx(0.2)

        @pytest.mark.parametrize(
            "threshold, expected_success",
            [
                (0.4, True),
                (0.5, True),
                (0.6, False),
            ],
        )
        @pytest.mark.asyncio
        async def test_threshold_can_be_configured(
            self, mock_native_model, threshold, expected_success
        ):
            metric = ContextRelevancyMetric(
                model=mock_native_model, threshold=threshold
            )
            metric.score = 0.5
            assert metric.is_successful() == expected_success

        @pytest.mark.asyncio
        async def test_strict_mode_overrides_threshold(
            self,
            mock_native_model,
        ):
            metric = ContextRelevancyMetric(
                model=mock_native_model, threshold=0.4, strict_mode=True
            )
            metric.score = 1.0
            assert metric.is_successful() is True

            metric.score = 0.999
            assert metric.is_successful() is False

        @pytest.mark.asyncio
        async def test_includes_reason_by_default(
            self,
            mock_native_model,
            mock_test_case,
            sample_reason,
        ):
            metric = ContextRelevancyMetric(model=mock_native_model)
            await metric.a_measure(mock_test_case)
            assert metric.reason == sample_reason.reason

        @pytest.mark.asyncio
        async def test_reason_none_when_include_reason_false(
            self,
            mock_native_model,
            mock_test_case,
            sample_truths,
            sample_information_needs,
            sample_verdicts,
        ):
            mock_native_model.a_generate.side_effect = [
                (sample_truths, 0.1),
                (sample_information_needs, 0.1),
                (sample_verdicts, 0.1),
            ]

            metric = ContextRelevancyMetric(
                model=mock_native_model, include_reason=False
            )
            await metric.a_measure(mock_test_case)

            assert metric.reason is None

        @pytest.mark.asyncio
        async def test_constructs_verbose_logs_when_verbose_mode(
            self,
            mock_native_model,
            mock_test_case,
            sample_truths,
            sample_information_needs,
            sample_verdicts,
            sample_reason,
        ):
            metric = ContextRelevancyMetric(
                model=mock_native_model, verbose_mode=True, include_reason=True
            )

            with patch(
                "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.context_relevancy.context_relevancy.construct_verbose_logs"
            ) as mock_logs:
                mock_logs.return_value = "mocked_verbose_logs"
                await metric.a_measure(mock_test_case)

                expected_steps = [
                    f"Truths: {prettify_list(sample_truths.truths)}",
                    f"Information Needs:\n{prettify_list(sample_information_needs.information_needs)}",
                    f"Verdicts:\n{prettify_list(sample_verdicts.verdicts)}",
                    f"Score: {metric.score}\nReason: {sample_reason.reason}",
                ]
                assert metric.verbose_logs == "mocked_verbose_logs"
                mock_logs.assert_called_once_with(metric, steps=expected_steps)

        @pytest.mark.asyncio
        async def test_non_native_model_base_behaviour(
            self,
            mock_non_native_model,
            mock_test_case,
            sample_truths,
            sample_information_needs,
            sample_verdicts,
            sample_reason,
        ):
            mock_non_native_model.a_generate.side_effect = [
                sample_truths,
                sample_information_needs,
                sample_verdicts,
                sample_reason,
            ]

            metric = ContextRelevancyMetric(model=mock_non_native_model)
            score = await metric.a_measure(mock_test_case)

            assert score == 0.5
            assert metric.reason == sample_reason.reason
            assert metric.evaluation_cost is None

        @pytest.mark.asyncio
        async def test_non_native_model_json_fallback(
            self, mock_non_native_model, mock_test_case
        ):
            truth_json = json.dumps({"truths": [{"context": "ctx", "facts": ["f1"]}]})
            info_needs_json = json.dumps({"information_needs": ["need"]})
            verdicts_json = json.dumps(
                {"verdicts": [{"verdict": "yes", "reason": "ok"}]}
            )
            reason_json = json.dumps({"reason": "fallback"})
            mock_non_native_model.a_generate.side_effect = [
                TypeError("schema fail"),
                truth_json,
                TypeError("schema fail"),
                info_needs_json,
                TypeError("schema fail"),
                verdicts_json,
                TypeError("schema fail"),
                reason_json,
            ]

            metric = ContextRelevancyMetric(model=mock_non_native_model)
            score = await metric.a_measure(mock_test_case)

            assert score == 1.0
            assert metric.reason == "fallback"

    def test_is_successful_returns_true_when_score_above_threshold(
        self, mock_native_model
    ):
        metric = ContextRelevancyMetric(model=mock_native_model, threshold=0.6)
        metric.score = 0.7
        assert metric.is_successful() is True

    def test_is_successful_returns_false_when_score_below_threshold(
        self, mock_native_model
    ):
        metric = ContextRelevancyMetric(model=mock_native_model, threshold=0.6)
        metric.score = 0.5
        assert metric.is_successful() is False

    def test_is_successful_returns_true_when_at_threshold(self, mock_native_model):
        metric = ContextRelevancyMetric(model=mock_native_model, threshold=0.6)
        metric.score = 0.6
        assert metric.is_successful() is True
