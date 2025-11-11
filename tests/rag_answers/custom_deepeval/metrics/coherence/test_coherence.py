from unittest.mock import AsyncMock, Mock, patch

import pytest
from deepeval.errors import MissingTestCaseParamsError
from deepeval.models import DeepEvalBaseLLM, GPTModel
from deepeval.test_case import LLMTestCase

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.coherence.coherence import (
    CoherenceMetric,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.coherence.schema import (
    CoherenceJudgement,
)


@pytest.fixture
def mock_test_case() -> LLMTestCase:
    return LLMTestCase(
        input="What support is available?",
        actual_output="Here is how to proceed.",
    )


@pytest.fixture
def mock_native_model() -> Mock:
    mock = Mock(spec=GPTModel)
    mock.get_model_name.return_value = "eu.amazon.nova-pro-v1:0"
    mock.a_generate = AsyncMock(
        return_value=(CoherenceJudgement(score=3, reason="Clear enough"), 0.2)
    )
    return mock


@pytest.fixture
def mock_non_native_model() -> Mock:
    mock = Mock(spec=DeepEvalBaseLLM)
    mock.get_model_name.return_value = "non-native-model"
    mock.a_generate = AsyncMock(
        return_value=CoherenceJudgement(score=4, reason="Reason text")
    )
    return mock


class TestCoherenceMetric:
    class TestAMeasure:
        @pytest.mark.asyncio
        async def test_missing_required_params_raises(self, mock_native_model: Mock):
            metric = CoherenceMetric(model=mock_native_model)
            invalid_case = LLMTestCase(input="query", actual_output=None)

            with pytest.raises(MissingTestCaseParamsError):
                await metric.a_measure(invalid_case)

        @pytest.mark.asyncio
        async def test_returns_normalised_score(
            self,
            mock_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            metric = CoherenceMetric(model=mock_native_model)

            score = await metric.a_measure(mock_test_case, _show_indicator=False)

            assert score == pytest.approx(0.5)  # (3 - 1) / 4
            assert metric.rubric_score == 3
            assert metric.reason == "Clear enough"

        @pytest.mark.asyncio
        async def test_with_native_model_tracks_cost(
            self,
            mock_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            metric = CoherenceMetric(model=mock_native_model)

            await metric.a_measure(mock_test_case, _show_indicator=False)

            assert metric.evaluation_cost == pytest.approx(0.2)

        @pytest.mark.asyncio
        async def test_with_non_native_model_skips_cost_tracking(
            self,
            mock_non_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            metric = CoherenceMetric(model=mock_non_native_model)

            await metric.a_measure(mock_test_case, _show_indicator=False)

            assert metric.evaluation_cost is None

        @pytest.mark.asyncio
        @patch(
            "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.coherence.coherence.metric_progress_indicator"
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
            mock_native_model: Mock,
            mock_test_case: LLMTestCase,
            set_show_progress: bool,
            expected_show_progress: bool,
        ):
            metric = CoherenceMetric(model=mock_native_model)

            await metric.a_measure(mock_test_case, _show_indicator=set_show_progress)

            mock_progress_indicator.assert_called_once_with(
                metric,
                async_mode=metric.async_mode,
                _show_indicator=expected_show_progress,
                _in_component=False,
            )

        @pytest.mark.asyncio
        async def test_constructs_verbose_logs_when_verbose_mode(
            self,
            mock_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            metric = CoherenceMetric(model=mock_native_model, verbose_mode=True)

            with patch(
                "govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.coherence.coherence.construct_verbose_logs"
            ) as mock_logs:
                mock_logs.return_value = "verbose"
                await metric.a_measure(mock_test_case)

                mock_logs.assert_called_once()
                steps = mock_logs.call_args.kwargs["steps"]
                assert any(step.startswith("Rubric Score:") for step in steps)
                assert any(step.startswith("Score:") for step in steps)
                assert any("Clear enough" in step for step in steps)
                assert metric.verbose_logs == "verbose"

        @pytest.mark.asyncio
        async def test_success_threshold_and_reason_toggle(
            self,
            mock_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            mock_native_model.a_generate = AsyncMock(
                return_value=(CoherenceJudgement(score=5, reason="Great"), 0.0)
            )
            metric = CoherenceMetric(
                model=mock_native_model,
                threshold=0.75,
                include_reason=False,
            )

            score = await metric.a_measure(mock_test_case)

            assert score == 1.0
            assert metric.success is True
            assert metric.reason is None

        @pytest.mark.asyncio
        async def test_non_native_model_json_fallback(
            self,
            mock_non_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            json_response = '{"score": 2, "reason": "Fallback path"}'
            mock_non_native_model.a_generate = AsyncMock(
                side_effect=[TypeError("no schema"), json_response]
            )
            metric = CoherenceMetric(model=mock_non_native_model)

            score = await metric.a_measure(mock_test_case)

            assert score == pytest.approx(0.25)
            assert metric.rubric_score == 2
            assert metric.reason == "Fallback path"
            assert (
                metric.evaluation_cost is None
            )  # non-native models skip cost tracking

            assert mock_non_native_model.a_generate.call_count == 2

        @pytest.mark.asyncio
        async def test_success_boundary_with_default_threshold(
            self,
            mock_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            mock_native_model.a_generate = AsyncMock(
                return_value=(CoherenceJudgement(score=4, reason="Almost there"), 0.0)
            )
            metric = CoherenceMetric(model=mock_native_model)

            await metric.a_measure(mock_test_case, _show_indicator=False)

            assert metric.score == pytest.approx(0.75)
            assert metric.success is False

            mock_native_model.a_generate = AsyncMock(
                return_value=(CoherenceJudgement(score=5, reason="Perfect"), 0.0)
            )
            metric = CoherenceMetric(model=mock_native_model)

            await metric.a_measure(mock_test_case, _show_indicator=False)

            assert metric.score == 1.0
            assert metric.success is True

        @pytest.mark.asyncio
        async def test_strict_mode_requires_perfect_score(
            self,
            mock_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            mock_native_model.a_generate = AsyncMock(
                return_value=(CoherenceJudgement(score=4, reason="Nearly"), 0.0)
            )
            metric = CoherenceMetric(model=mock_native_model, strict_mode=True)

            assert metric.threshold == 1.0
            await metric.a_measure(mock_test_case, _show_indicator=False)

            assert metric.score == pytest.approx(0.75)
            assert metric.success is False

            mock_native_model.a_generate = AsyncMock(
                return_value=(CoherenceJudgement(score=5, reason="Fully coherent"), 0.0)
            )
            metric = CoherenceMetric(model=mock_native_model, strict_mode=True)

            await metric.a_measure(mock_test_case, _show_indicator=False)

            assert metric.success is True

        @pytest.mark.asyncio
        async def test_score_is_clamped_into_range(
            self,
            mock_non_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            mock_non_native_model.a_generate = AsyncMock(
                return_value=CoherenceJudgement(score=999, reason="Out of bounds")
            )
            metric = CoherenceMetric(model=mock_non_native_model)

            score = await metric.a_measure(mock_test_case)

            assert score == 1.0
            assert metric.rubric_score == 5

        @pytest.mark.asyncio
        async def test_score_is_clamped_at_lower_bound(
            self,
            mock_non_native_model: Mock,
            mock_test_case: LLMTestCase,
        ):
            mock_non_native_model.a_generate = AsyncMock(
                return_value=CoherenceJudgement(score=0, reason="Too low")
            )
            metric = CoherenceMetric(model=mock_non_native_model)

            score = await metric.a_measure(mock_test_case)

            assert score == 0.0
            assert metric.rubric_score == 1

    class TestIsSuccessful:
        def test_is_successful_returns_true_when_score_above_threshold(
            self, mock_native_model
        ):
            metric = CoherenceMetric(model=mock_native_model, threshold=0.8)
            metric.score = 1.0
            assert metric.is_successful() is True

        def test_is_successful_returns_true_when_at_threshold(self, mock_native_model):
            metric = CoherenceMetric(model=mock_native_model, threshold=0.8)
            metric.score = 0.75
            assert metric.is_successful() is False

        def test_is_successful_returns_false_when_score_below_threshold(
            self, mock_native_model
        ):
            metric = CoherenceMetric(model=mock_native_model, threshold=0.8)
            metric.score = 0.5
            assert metric.is_successful() is False
