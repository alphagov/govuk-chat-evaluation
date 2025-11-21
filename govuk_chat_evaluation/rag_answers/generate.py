import asyncio
from pathlib import Path

from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output
from .data_models import (
    GenerateInput,
    EvaluationTestCase,
    StructuredContext,
)
from collections import defaultdict


def generate_and_write_dataset(input_path: Path, provider: str, output_dir: Path):
    models = jsonl_to_models(Path(input_path), GenerateInput)
    models_with_unique_ids = ensure_model_ids_are_unique(models)
    generated = generate_inputs_to_evaluation_test_cases(provider, models_with_unique_ids)

    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_test_cases(
    provider: str, generate_inputs: list[GenerateInput]
) -> list[EvaluationTestCase]:
    """Asynchronously run rake tasks for each GenerateInput instance to
    generate models that can be evaluated"""

    async def generate_input_to_evaluation_test_case(input: GenerateInput):
        env = {"INPUT": input.question}
        result = await run_rake_task(
            f"evaluation:generate_rag_structured_answer_response[{provider}]",
            env,
        )

        # Extract context from result
        structured_contexts = result["sources"]
        structured_contexts = [
            StructuredContext(**ctx["chunk"]) for ctx in structured_contexts
        ]

        # TODO: this will need more data fields and may well want to validate
        # aspects of the returned data rather than just using the JSON directly
        return EvaluationTestCase(
            id=input.id,
            question=input.question,
            ideal_answer=input.ideal_answer,
            llm_answer=result["message"],
            structured_contexts=structured_contexts,
        )

    return asyncio.run(
        generate_dataset(generate_inputs, generate_input_to_evaluation_test_case)
    )


def ensure_model_ids_are_unique(
    inputs: list[GenerateInput],
) -> list[GenerateInput]:
    ids = [input.id for input in inputs]
    if len(ids) == len(set(ids)):
        return inputs

    inputs_by_id = defaultdict(list)

    for input in inputs:
        inputs_by_id[input.id].append(input)

    for _id, grouped_inputs in inputs_by_id.items():
        if len(grouped_inputs) > 1:
            for index, input in enumerate(grouped_inputs[1:], start=1):
                input.id = f"{input.id}-duplicate-{index}"

    return inputs
