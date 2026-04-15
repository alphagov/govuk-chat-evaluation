import asyncio
from pathlib import Path

from pydantic import BaseModel

from .evaluate import EvaluationResult, SearchResult
from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output
from typing import Optional


class GenerateInput(BaseModel):
    question: str
    expected_exact_paths: list[str]
    expected_chunk_uids: list[str]
    expected_opensearch_index: Optional[str] = None


def generate_and_write_dataset(input_path: Path, output_dir: Path):
    models = jsonl_to_models(Path(input_path), GenerateInput)
    generated = generate_inputs_to_evaluation_results(models)
    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_results(
    generate_inputs: list[GenerateInput],
) -> list[EvaluationResult]:
    """Asynchronously run rake tasks for each GenerateInput instance to
    generate a result"""

    async def generate_input_to_evaluation_result(input: GenerateInput):
        env = {"INPUT": input.question}
        if input.expected_opensearch_index:
            env["OPENSEARCH_INDEX"] = input.expected_opensearch_index

        result = await run_rake_task(
            "evaluation:search_results_for_question",
            env,
        )
        actual_opensearch_index = result["opensearch_index"]
        results = result["results"]
        exact_paths_chunks_and_scores = [
            SearchResult(
                exact_path=item["exact_path"],
                chunk_uid=item["chunk_uid"],
                weighted_score=item["weighted_score"],
                semantic_score=item["score"],
            )
            for item in results
        ]

        return EvaluationResult(
            question=input.question,
            expected_exact_paths=input.expected_exact_paths,
            expected_chunk_uids=input.expected_chunk_uids,
            expected_opensearch_index=input.expected_opensearch_index,
            actual_search_results=exact_paths_chunks_and_scores,
            actual_opensearch_index=actual_opensearch_index,
        )

    return asyncio.run(
        generate_dataset(generate_inputs, generate_input_to_evaluation_result)
    )
