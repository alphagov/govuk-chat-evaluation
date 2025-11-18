from pathlib import Path
from typing import cast
from functools import cached_property
import pandas as pd

from deepeval.metrics import BaseMetric
from deepeval.evaluate.configs import (
    AsyncConfig,
    DisplayConfig,
    CacheConfig,
    ErrorConfig,
)

from .deepeval_evaluate import (
    run_deepeval_evaluation,
    convert_deepeval_output_to_evaluation_results,
)
from ..file_system import jsonl_to_models
from .data_models import EvaluationTestCase, TaskConfig
from .deepeval_evaluate import (
    EvaluationResult,
)
import logging


display_config = DisplayConfig(
    verbose_mode=False,
    show_indicator=True,
    print_results=False,
)

async_config = AsyncConfig(
    max_concurrent=40,
    throttle_value=5,
)

cache_config = CacheConfig(use_cache=False, write_cache=False)

error_config = ErrorConfig(
    ignore_errors=True,
)


# would expect we need to pass config object through if that has metrics configuration
def evaluate_and_output_results(
    output_dir: Path, evaluation_data_path: Path, evaluation_config: TaskConfig
):
    """
    Function to run the evaluation, aggregate the results, and export them to files.

    Args:
        output_dir: The directory to save the evaluation results.
        evaluation_data_path: Path to the JSONL file containing the evaluation data.
        evaluation_config: Configuration for the evaluation.
    """

    models = jsonl_to_models(evaluation_data_path, EvaluationTestCase)

    if not models:
        logging.error("\nThere is no data to evaluate")
        return

    evaluation_outputs = run_deepeval_evaluation(
        cases=[model.to_llm_test_case() for model in models],
        metrics=cast(list[BaseMetric], evaluation_config.metric_instances()),
        n_runs=evaluation_config.n_runs,
        display_config=display_config,
        async_config=async_config,
        cache_config=cache_config,
        error_config=error_config,
        output_dir=output_dir,
    )

    evaluation_results = convert_deepeval_output_to_evaluation_results(
        evaluation_outputs
    )

    _log_metric_errors(evaluation_results)

    aggregation = AggregatedResults(evaluation_results)

    # calculate aggregated results and exports results to CSV files
    aggregation.export_to_csvs(output_dir)

    logging.info("Evaluation Results:")
    logging.info(aggregation.summary)


class AggregatedResults:
    def __init__(self, evaluation_results: list[EvaluationResult]):
        self.evaluation_results = evaluation_results

    @cached_property
    def per_input_metric_averages(self) -> pd.DataFrame:
        """
        Computes average metric scores per test input.

        Returns:
            DataFrame with rows as test names and columns as metrics.
        """
        # flatten the evaluation results
        data = []

        for eval_result in self.evaluation_results or []:
            for evaluation_output in eval_result.run_metric_outputs:
                data.append(
                    {
                        "name": eval_result.name,
                        "input": eval_result.input,
                        "metric": evaluation_output.metric,
                        "score": evaluation_output.score,
                        "error": evaluation_output.error,
                    }
                )

        df = pd.DataFrame(data)

        return (
            df.groupby(["name", "input", "metric"])["score"]
            .agg(["mean", "std", "count"])
            .unstack()
            .reset_index()
            .rename(columns={"count": "n_datapoints"})
        )

    @cached_property
    def summary(self) -> pd.DataFrame:
        """
        Summary statistics across all inputs: median, mean, std per metric.

        Returns:
            DataFrame with metric as index and stats as columns.
        """

        mean_df = self.per_input_metric_averages["mean"]

        return pd.DataFrame(
            {
                "median": mean_df.median(),
                "mean": mean_df.mean(),
                "std": mean_df.std(),
                "n_datapoints": self.per_input_metric_averages["n_datapoints"].sum(),
            }
        )

    def export_to_csvs(self, output_dir: Path) -> None:
        """
        Exports per-input and summary metric statistics to CSV files.
        """
        pd.DataFrame(self.evaluation_results).to_csv(output_dir / "tidy_results.csv")
        self.per_input_metric_averages.to_csv(output_dir / "results_per_input.csv")
        self.summary.to_csv(output_dir / "results_summary.csv")


def _log_metric_errors(evaluation_results: list[EvaluationResult]) -> None:
    """Emit warnings for metrics that errored so problems.log records them.

    DeepEval sets `error` on metric data when `ignore_errors=True`. We surface
    that here without failing the run.
    """
    for er in evaluation_results:
        for out in er.run_metric_outputs:
            if out.error:
                logging.warning(
                    "Metric error (name=%s, metric=%s, run=%s): %s",
                    er.name,
                    out.metric,
                    out.run,
                    out.error,
                )
