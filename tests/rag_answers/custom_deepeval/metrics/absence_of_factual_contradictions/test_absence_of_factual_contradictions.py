import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.errors import MissingTestCaseParamsError
from deepeval.utils import prettify_list

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.absence_of_factual_contradictions.absence_of_factual_contradictions import (
    AbsenceOfFactualContradictions,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.absence_of_factual_contradictions.schema import (
    TruthCollection,
    ClaimCollection,
    Verdict,
    VerdictCollection,
    ScoreReason,
)


@pytest.fixture
def mock_native_model(sample_truths, sample_claims, sample_verdicts, sample_reason):
    mock = Mock(spec=GPTModel)
    mock.get_model_name.return_value = "eu.amazon.nova-pro-v1:0"
    mock.a_generate = AsyncMock()
    mock.a_generate.side_effect = [
        (sample_truths, 0.1),
        (sample_claims, 0.1),
        (sample_verdicts, 0.1),
        (sample_reason, 0.1),
    ]
    return mock


@pytest.fixture
def mock_non_native_model(sample_truths, sample_claims, sample_verdicts, sample_reason):
    mock = Mock(spec=DeepEvalBaseLLM)
    mock.get_model_name.return_value = "non-native-model"
    mock.a_generate = AsyncMock()
    mock.a_generate.side_effect = [
        sample_truths,
        sample_claims,
        sample_verdicts,
        sample_reason,
    ]
    return mock


@pytest.fixture
def mock_test_case():
    return LLMTestCase(
        input="What did Einstein win?",
        actual_output="Einstein won the Nobel Prize.",
        expected_output="Einstein won the Nobel Prize in 1921.",
    )


@pytest.fixture
def sample_truths():
    return TruthCollection(truths=["Einstein won the Nobel Prize in 1921."])


@pytest.fixture
def sample_claims():
    return ClaimCollection(claims=["Einstein won the Nobel Prize."])


@pytest.fixture
def sample_verdicts():
    return VerdictCollection(
        verdicts=[
            Verdict(verdict="yes"),
            Verdict(verdict="no", reason="Incomplete information."),
        ]
    )


@pytest.fixture
def sample_reason():
    return ScoreReason(
        reason="The actual output is mostly correct but missing the year."
    )


class TestAbsenceOfFactualContradictions:
    class TestAMeasure:
        @pytest.mark.asyncio
        async def test_invalid_params_raise_error(self, mock_native_model):
            metric = AbsenceOfFactualContradictions(model=mock_native_model)
            # Missing expected_output
            invalid_case = LLMTestCase(input="question", actual_output="answer")
            with pytest.raises(MissingTestCaseParamsError, match="expected_output"):
                await metric.a_measure(invalid_case)

        @pytest.mark.asyncio
        @patch(
            "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.absence_of_factual_contradictions.absence_of_factual_contradictions.metric_progress_indicator"
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
            metric = AbsenceOfFactualContradictions(model=mock_native_model)
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
            metric = AbsenceOfFactualContradictions(
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
        ):
            metric = AbsenceOfFactualContradictions(model=mock_native_model)
            await metric.a_measure(mock_test_case)

            # 4 calls * 0.1 cost = 0.4
            assert metric.evaluation_cost == 0.4

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
            metric = AbsenceOfFactualContradictions(
                model=mock_native_model, threshold=threshold
            )
            metric.score = 0.5
            assert metric.is_successful() == expected_success

        @pytest.mark.asyncio
        async def test_strict_mode_overrides_threshold(
            self,
            mock_native_model,
        ):
            metric = AbsenceOfFactualContradictions(
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
            metric = AbsenceOfFactualContradictions(model=mock_native_model)
            await metric.a_measure(mock_test_case)
            assert metric.reason == sample_reason.reason

        @pytest.mark.asyncio
        async def test_reason_none_when_include_reason_false(
            self,
            mock_native_model,
            mock_test_case,
            sample_truths,
            sample_claims,
            sample_verdicts,
        ):
            mock_native_model.a_generate.side_effect = [
                (sample_truths, 0.1),
                (sample_claims, 0.1),
                (sample_verdicts, 0.1),
            ]

            metric = AbsenceOfFactualContradictions(
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
            sample_claims,
            sample_verdicts,
            sample_reason,
        ):
            metric = AbsenceOfFactualContradictions(
                model=mock_native_model, verbose_mode=True, include_reason=True
            )

            with patch(
                "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.absence_of_factual_contradictions.absence_of_factual_contradictions.construct_verbose_logs"
            ) as mock_logs:
                mock_logs.return_value = "mocked_verbose_logs"
                await metric.a_measure(mock_test_case)

                expected_steps = [
                    f"Truths:\n{prettify_list(sample_truths.truths)}",
                    f"Claims:\n{prettify_list(sample_claims.claims)}",
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
            sample_claims,
            sample_verdicts,
            sample_reason,
        ):
            mock_non_native_model.a_generate.side_effect = [
                sample_truths,
                sample_claims,
                sample_verdicts,
                sample_reason,
            ]

            metric = AbsenceOfFactualContradictions(model=mock_non_native_model)
            score = await metric.a_measure(mock_test_case)

            assert score == 0.5
            assert metric.reason == sample_reason.reason
            assert metric.evaluation_cost is None

        @pytest.mark.asyncio
        async def test_non_native_model_json_fallback(
            self, mock_non_native_model, mock_test_case
        ):
            truths_json = json.dumps({"truths": ["t1"]})
            claims_json = json.dumps({"claims": ["c1"]})
            verdicts_json = json.dumps(
                {"verdicts": [{"verdict": "yes", "reason": None}]}
            )
            reason_json = json.dumps({"reason": "fallback reason"})

            mock_non_native_model.a_generate.side_effect = [
                TypeError("schema fail"),
                truths_json,
                TypeError("schema fail"),
                claims_json,
                TypeError("schema fail"),
                verdicts_json,
                TypeError("schema fail"),
                reason_json,
            ]

            metric = AbsenceOfFactualContradictions(model=mock_non_native_model)
            score = await metric.a_measure(mock_test_case)

            assert score == 1.0
            assert metric.reason == "fallback reason"

    def test_is_successful_returns_true_when_score_above_threshold(
        self, mock_native_model
    ):
        metric = AbsenceOfFactualContradictions(model=mock_native_model, threshold=0.6)
        metric.score = 0.7
        assert metric.is_successful() is True

    def test_is_successful_returns_false_when_score_below_threshold(
        self, mock_native_model
    ):
        metric = AbsenceOfFactualContradictions(model=mock_native_model, threshold=0.6)
        metric.score = 0.5
        assert metric.is_successful() is False

    def test_is_successful_returns_true_when_at_threshold(self, mock_native_model):
        metric = AbsenceOfFactualContradictions(model=mock_native_model, threshold=0.6)
        metric.score = 0.6
        assert metric.is_successful() is True
