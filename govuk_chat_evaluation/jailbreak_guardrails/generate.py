import asyncio
import json
import logging
from pathlib import Path

from pydantic import BaseModel

from ..dataset_generation import generate_dataset, run_rake_task
from ..file_system import jsonl_to_models, write_generated_to_output
from .evaluate import EvaluationResult

logger = logging.getLogger(__name__)


class GenerateInput(BaseModel):
    question: str
    expected_outcome: bool


def generate_and_write_dataset(
    input_path: Path,
    claude_generation_model: str | None,
    output_dir: Path,
):
    models = jsonl_to_models(input_path, GenerateInput)
    generated = generate_inputs_to_evaluation_results(claude_generation_model, models)
    return write_generated_to_output(output_dir, generated)


def generate_inputs_to_evaluation_results(
    claude_generation_model: str | None,
    generate_inputs: list[GenerateInput],
) -> list[EvaluationResult]:
    """Asynchronously run rake tasks for each GenerateInput instance to
    generate a result"""

    async def generate_input_to_evaluation_result(input: GenerateInput):
        env = {"INPUT": input.question}
        if claude_generation_model:
            env["BEDROCK_CLAUDE_JAILBREAK_GUARDRAILS_MODEL"] = claude_generation_model

        result = await run_rake_task(
            "evaluation:generate_jailbreak_guardrail_response",
            env,
        )

        jailbreak_guardrails_status = result.get("jailbreak_guardrails_status")

        if jailbreak_guardrails_status == "error":
            parsed_jailbreak_llm_response = json.loads(result["llm_responses"])[
                "jailbreak_guardrails"
            ]
            invalid_llm_response = parsed_jailbreak_llm_response["content"][0]["text"]

            logger.warning(
                f"Invalid response for {input.question!r}, returned: {invalid_llm_response!r}"
            )
            return None

        actual_outcome = "pass" not in jailbreak_guardrails_status
        return EvaluationResult(
            question=input.question,
            expected_outcome=input.expected_outcome,
            actual_outcome=actual_outcome,
            model=result["metrics"]["jailbreak_guardrails"]["model"],
        )

    return asyncio.run(
        generate_dataset(generate_inputs, generate_input_to_evaluation_result)
    )
