import logging
from types import MethodType
from typing import Any, Optional, TypeVar

from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM


ModelT = TypeVar("ModelT", bound=DeepEvalBaseLLM)


def attach_invalid_json_retry_to_model(
    model: ModelT,
    max_attempts: int = 3,
) -> ModelT:
    """Wrap model.a_generate to retry on DeepEval's invalid JSON error."""
    if max_attempts <= 0:
        msg = "max_attempts must be a positive integer to enable invalid JSON retries"
        raise ValueError(msg)

    original = model.a_generate

    async def _retrying_a_generate(
        self: DeepEvalBaseLLM,
        prompt: str,
        schema: Optional[BaseModel] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        for attempt in range(max_attempts):
            try:
                return await original(prompt, schema, *args, **kwargs)
            except ValueError as exc:
                message = str(exc).lower()
                last_attempt = attempt == max_attempts - 1
                if "invalid json" not in message or last_attempt:
                    raise

                attempts_remaining = max_attempts - attempt - 1
                if attempts_remaining > 1:
                    detail = f"{attempts_remaining} attempts left"
                else:
                    detail = "last attempt remaining"

                logging.warning(
                    "LLM judge emitted invalid JSON; retrying (%s)",
                    detail,
                )

    model.a_generate = MethodType(_retrying_a_generate, model)
    return model
