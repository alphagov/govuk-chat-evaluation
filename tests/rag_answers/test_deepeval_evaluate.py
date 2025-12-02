import pytest
from unittest.mock import MagicMock

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import BaseMetric
from deepeval.evaluate.configs import (
    AsyncConfig,
    DisplayConfig,
    CacheConfig,
    ErrorConfig,
)

from govuk_chat_evaluation.file_system import jsonl_to_models
from govuk_chat_evaluation.rag_answers.data_models import (
    EvaluationTestCase,
    TaskConfig,
)
from govuk_chat_evaluation.rag_answers.deepeval_evaluate import (
    EvaluationResult,
    run_deepeval_evaluation,
    convert_deepeval_output_to_evaluation_results,
)
from tests.conftest import assert_mock_call_matches_signature
import json
import logging
from deepeval.test_run import TestRun as DeepevalTestRun


@pytest.mark.usefixtures("mock_deepeval_evaluate")
class TestRunDeepEvalEvaluation:
    @pytest.fixture
    def mock_test_cases(self, mock_input_data):
        models = jsonl_to_models(mock_input_data, EvaluationTestCase)
        return [m.to_llm_test_case() for m in models]

    @pytest.fixture
    def mock_metrics(self):
        metric1 = MagicMock(spec=BaseMetric)
        metric1.name = "faithfulness"
        metric1.threshold = 0.5
        metric1.async_mode = False

        metric2 = MagicMock(spec=BaseMetric)
        metric2.name = "bias"
        metric2.threshold = 0.5
        metric2.async_mode = False

        return [metric1, metric2]

    @pytest.fixture
    def mock_task_config(self, mock_input_data, mock_metrics):
        config = MagicMock(spec=TaskConfig)
        config.metric_instances.return_value = mock_metrics
        return config

    def test_runs_evaluate_defaults_to_1_time(
        self,
        mock_test_cases,
        mock_metrics,
        mock_task_config,
        mock_deepeval_evaluate,
        mock_project_root,
    ):
        run_deepeval_evaluation(mock_test_cases, mock_task_config, mock_project_root)
        assert_mock_call_matches_signature(mock_deepeval_evaluate, deepeval_evaluate)
        mock_deepeval_evaluate.assert_called_once_with(
            test_cases=mock_test_cases, metrics=mock_metrics
        )

    def test_runs_evaluate_n_times(
        self,
        mock_test_cases,
        mock_metrics,
        mock_task_config,
        mock_deepeval_evaluate,
        mock_project_root,
    ):
        run_deepeval_evaluation(
            mock_test_cases, mock_task_config, mock_project_root, n_runs=2
        )
        assert_mock_call_matches_signature(mock_deepeval_evaluate, deepeval_evaluate)
        mock_deepeval_evaluate.assert_called_with(
            test_cases=mock_test_cases, metrics=mock_metrics
        )

        assert mock_deepeval_evaluate.call_count == 2

    def test_returns_a_list_for_each_n_runs(
        self, mock_test_cases, mock_task_config, mock_project_root
    ):
        results = run_deepeval_evaluation(
            mock_test_cases, mock_task_config, mock_project_root, n_runs=2
        )
        assert len(results) == 2

    def test_accepts_deepeval_options(
        self,
        mock_test_cases,
        mock_metrics,
        mock_task_config,
        mock_deepeval_evaluate,
        mock_project_root,
    ):
        run_deepeval_evaluation(
            mock_test_cases,
            mock_task_config,
            mock_project_root,
            display_config=DisplayConfig(print_results=True),
            async_config=AsyncConfig(max_concurrent=10),
            cache_config=CacheConfig(use_cache=True),
            error_config=ErrorConfig(ignore_errors=False),
        )

        assert_mock_call_matches_signature(mock_deepeval_evaluate, deepeval_evaluate)

        mock_deepeval_evaluate.assert_called_with(
            test_cases=mock_test_cases,
            metrics=mock_metrics,
            display_config=DisplayConfig(print_results=True),
            async_config=AsyncConfig(max_concurrent=10),
            cache_config=CacheConfig(use_cache=True),
            error_config=ErrorConfig(ignore_errors=False),
        )

    def test_evaluate_and_output_results_writes_deepeval_test_run(
        self, mock_test_cases, mock_task_config, mocker, mock_project_root, caplog
    ):
        test_run = mocker.create_autospec(DeepevalTestRun, instance=True)
        test_run.model_dump.return_value = {"id": "test-run-123"}
        mocker.patch(
            "govuk_chat_evaluation.rag_answers.deepeval_evaluate.global_test_run_manager.get_test_run",
            return_value=test_run,
        )
        caplog.set_level(logging.INFO)

        n_runs = 2
        run_deepeval_evaluation(
            mock_test_cases, mock_task_config, mock_project_root, n_runs=n_runs
        )

        for i in range(n_runs):
            output_path = mock_project_root / f"deepeval_test_run_{i + 1}.json"
            assert output_path.exists()

            with output_path.open() as f:
                data = json.load(f)
                assert data == {"id": "test-run-123"}

            assert (
                f"Run {i + 1} done. written to {output_path.relative_to(mock_project_root)}"
                in caplog.text
            )

        assert test_run.model_dump.call_count == n_runs
        test_run.model_dump.assert_called_with(by_alias=True, exclude_none=True)

    def test_evaluate_and_output_results_raises_if_no_test_run(
        self,
        mock_test_cases,
        mock_task_config,
        mocker,
        mock_project_root,
    ):
        mocker.patch(
            "govuk_chat_evaluation.rag_answers.deepeval_evaluate.global_test_run_manager.get_test_run",
            return_value=None,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_deepeval_evaluation(
                mock_test_cases, mock_task_config, mock_project_root, n_runs=1
            )

        assert str(exc_info.value) == "DeepEval test run not found for run 1"


class TestConvertDeepEvalOutput:
    def test_convert_empty_results(self):
        results = convert_deepeval_output_to_evaluation_results([])
        assert results == []

    def test_converts_to_evaluation_results(self, mock_deepeval_results):
        results = convert_deepeval_output_to_evaluation_results(mock_deepeval_results)

        assert all(isinstance(item, EvaluationResult) for item in results)

        # each item should have the number of metrics multiplied by runs
        metrics_tested = len(mock_deepeval_results[0][0].metrics_data)
        runs = len(mock_deepeval_results)
        expected_metrics = metrics_tested * runs

        assert all(len(item.run_metric_outputs) == expected_metrics for item in results)

    def test_with_none_retrieval_context(self, mock_deepeval_results):
        # modify test data to have None for retrieval_context
        mock_deepeval_results[0][0].retrieval_context = None

        results = convert_deepeval_output_to_evaluation_results(
            [mock_deepeval_results[0]]
        )

        assert results[0].retrieval_context == []
