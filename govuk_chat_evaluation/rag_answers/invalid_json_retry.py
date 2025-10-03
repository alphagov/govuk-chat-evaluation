import logging
from types import MethodType
from typing import Any, Optional, Protocol, TypeVar

from pydantic import BaseModel


class SupportsAGenerate(Protocol):
    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...


ModelT = TypeVar("ModelT", bound=SupportsAGenerate)


def attach_invalid_json_retry_to_model(
    model: ModelT,
    max_attempts: int = 3,
) -> ModelT:
    """Wrap model.a_generate to retry on DeepEval's invalid JSON error."""
    if max_attempts <= 1:
        return model

    original = model.a_generate

    async def _retrying_a_generate(
        self: SupportsAGenerate,
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
                logging.warning(
                    "LLM judge emitted invalid JSON; retrying (%s attempts left)",
                    attempts_remaining,
                )

    model.a_generate = MethodType(_retrying_a_generate, model)
    return model
