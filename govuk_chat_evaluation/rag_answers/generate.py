import asyncio
from pathlib import Path

from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output
from .data_models import (
    GenerateInput,
    EvaluationTestCase,
    StructuredContext,
)
from govuk_chat_evaluation.rag_answers.handle_model_id_collisions import (
    ensure_unique_model_ids,
)
from typing import Optional


def generate_and_write_dataset(
    input_path: Path,
    provider: str,
    claude_generation_model: Optional[str],
    output_dir: Path,
):
    models = jsonl_to_models(Path(input_path), GenerateInput)
    ensure_unique_model_ids(models)
    generated = generate_inputs_to_evaluation_test_cases(
        provider, claude_generation_model, models
    )

    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_test_cases(
    provider: str,
    claude_generation_model: Optional[str],
    generate_inputs: list[GenerateInput],
) -> list[EvaluationTestCase]:
    """Asynchronously run rake tasks for each GenerateInput instance to
    generate models that can be evaluated"""

    async def generate_input_to_evaluation_test_case(input: GenerateInput):
        env = {"INPUT": input.question}
        if claude_generation_model:
            env["BEDROCK_CLAUDE_STRUCTURED_ANSWER_COMPOSER_MODEL"] = (
                claude_generation_model
            )

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
