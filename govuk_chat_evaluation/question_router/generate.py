import asyncio
from pathlib import Path

from pydantic import BaseModel
from typing import Optional

from .evaluate import EvaluationResult
from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output


class GenerateInput(BaseModel):
    question: str
    expected_outcome: str


def generate_and_write_dataset(
    input_path: Path,
    claude_generation_model: Optional[str],
    output_dir: Path,
):
    models = jsonl_to_models(Path(input_path), GenerateInput)
    generated = generate_inputs_to_evaluation_results(claude_generation_model, models)
    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_results(
    claude_generation_model: Optional[str],
    generate_inputs: list[GenerateInput],
) -> list[EvaluationResult]:
    """Asynchronously run rake tasks for each GenerateInput instance to
    generate a result"""

    async def generate_input_to_evaluation_result(input: GenerateInput):
        env = {"INPUT": input.question}
        if claude_generation_model:
            env["BEDROCK_CLAUDE_QUESTION_ROUTER_MODEL"] = claude_generation_model

        result = await run_rake_task(
            "evaluation:generate_question_routing_response",
            env,
        )

        return EvaluationResult(
            question=input.question,
            expected_outcome=input.expected_outcome,
            actual_outcome=result["question_routing_label"],
            confidence_score=result["question_routing_confidence_score"],
            answer=result["message"],
            model=result["metrics"]["question_routing"]["model"],
        )

    return asyncio.run(
        generate_dataset(generate_inputs, generate_input_to_evaluation_result)
    )
