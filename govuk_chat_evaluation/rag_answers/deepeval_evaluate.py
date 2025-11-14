from collections import defaultdict
from pydantic.dataclasses import dataclass
from typing import Optional

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.types import TestResult
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from ..timing import log_task_duration
import logging


@dataclass
class RunMetricOutput:
    run: int
    metric: str
    score: float | None = None
    cost: float | None = None
    reason: str | None = None
    success: bool | None = None
    error: str | None = None


@dataclass
class EvaluationResult:
    name: str
    input: str
    actual_output: str
    retrieval_context: list[str]
    run_metric_outputs: list[RunMetricOutput]
    expected_output: Optional[str] = None


def run_deepeval_evaluation(
    cases: list[LLMTestCase], metrics: list[BaseMetric], n_runs: int = 1, **kwargs
) -> list[list[TestResult]]:
    """ "
    Run the DeepEval evaluation on the given models and metrics

    Args:
        cases : List of test cases to evaluate
        metrics : List of metrics to use for evaluation
        n_runs : Number of runs to perform for the evaluation
        **kwargs: Additional arguments to pass to the deepeval.evaluation function

    Returns:
        Evaluation results grouped by run

    """

    with log_task_duration("Running DeepEval Evaluation"):
        logging.info("Running DeepEval evaluation")

        all_evaluation_runs = []

        for i in range(n_runs):
            logging.info(f"Running evaluation iteration {i + 1}/{n_runs}...")

            evaluation_run = deepeval_evaluate(
                test_cases=cases,
                metrics=metrics,
                **kwargs,  # pass additional arguments dynamically
            )

            all_evaluation_runs.append(
                evaluation_run.test_results
            )  # Store results per run

    logging.info("DeepEval evaluation complete")

    return all_evaluation_runs


def convert_deepeval_output_to_evaluation_results(
    all_runs: list[list[TestResult]],
) -> list[EvaluationResult]:
    """
    Convert the results from DeepEval into a more structured format.

    Intermediate steps:
    1. Group the results by input and run index.
                {
            'input_1': {
                0: [TestResult],
                1: [TestResult],
                2: [TestResult,],
            },
            'input_2': {
                0: [TestResult],
                1: [TestResult],
                2: [TestResult],
            },
        }
    2. For each input, create an EvaluationResult object containing the input, actual output, expected output, and the evaluation results for each run.
        [
            EvaluationResult(
                name='input_1',
                input='input_1',
                actual_output='actual_output_1',
                expected_output='expected_output_1',
                retrieval_context=['context_1'],
                evaluation_results=[
                    RunMetricOutput(
                        run=0,
                        metric='metric_1',
                        score=0.9,
                        reason='reason_1',
                        cost=0.5,
                        success=True
                    ),
                    RunMetricOutput(
                        run=1,
                        metric='metric_2',
                        score=0.8,
                        reason='reason_2',
                        cost=0.6,
                        success=True
                    )
                ]
            ),
            EvaluationResult(
                name='input_2',
                ...)
                ]

    3. Return a list of EvaluationResult objects.

    Args:
        all_runs : List of evaluation runs, each containing a list of TestResult objects, one object per test case, the ouput of run_deepeval_evaluation
    """
    grouped_by_input_and_run = defaultdict(lambda: defaultdict(list))

    for run_idx, run in enumerate(all_runs):  # all_runs = List[List[TestResult]]
        for result in run:
            grouped_by_input_and_run[result.name][run_idx].append(result)

    aggregated_results: list[EvaluationResult] = []

    for _input_name, run_results in grouped_by_input_and_run.items():
        evaluation_outputs: list[RunMetricOutput] = []

        # taking the first TestResult for each input to extract static info
        sample_result = run_results[0][0]

        for run_idx, results in run_results.items():
            for result in results:
                for metric_data in result.metrics_data:
                    evaluation_outputs.append(
                        RunMetricOutput(
                            run=run_idx,
                            metric=metric_data.name,
                            score=metric_data.score,
                            reason=metric_data.reason,
                            cost=metric_data.evaluation_cost,
                            success=metric_data.success,
                            error=metric_data.error,
                        )
                    )

        aggregated_results.append(
            EvaluationResult(
                name=sample_result.name,
                input=sample_result.input,
                actual_output=sample_result.actual_output,
                expected_output=sample_result.expected_output,
                retrieval_context=sample_result.retrieval_context or [],
                run_metric_outputs=evaluation_outputs,
            )
        )

    return aggregated_results
