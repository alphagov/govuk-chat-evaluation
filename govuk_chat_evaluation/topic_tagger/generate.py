import asyncio
from pathlib import Path

from pydantic import BaseModel

from .evaluate import EvaluationResult
from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output


class GenerateInput(BaseModel):
    question: str
    expected_primary_topic: str
    expected_secondary_topic: str | None


def generate_and_write_dataset(input_path: Path, output_dir: Path):
    models = jsonl_to_models(input_path, GenerateInput)
    generated = generate_inputs_to_evaluation_results(models)
    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_results(
    generate_inputs: list[GenerateInput],
) -> list[EvaluationResult]:
    async def generate_input_to_evaluation_result(input: GenerateInput):
        env = {"INPUT": input.question}
        result = await run_rake_task(
            "evaluation:generate_topics_for_question",
            env,
        )

        if "primary_topic" in result:
            return EvaluationResult(
                question=input.question,
                expected_primary_topic=input.expected_primary_topic,
                actual_primary_topic=result.get("primary_topic"),
                expected_secondary_topic=input.expected_secondary_topic,
                actual_secondary_topic=result.get("secondary_topic"),
            )
        else:
            raise RuntimeError(f"Unexpected result structure {result!r}")

    return asyncio.run(
        generate_dataset(generate_inputs, generate_input_to_evaluation_result)
    )
