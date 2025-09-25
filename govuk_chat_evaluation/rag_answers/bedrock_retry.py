import logging
from types import MethodType
from typing import Any, Optional

from pydantic import BaseModel

from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel


def attach_retry_to_bedrock_model(
    model: AmazonBedrockModel,
    max_attempts: int = 3,
    error_substring: str = "invalid json",
) -> AmazonBedrockModel:
    """Wrap model.a_generate to retry on DeepEval's invalid JSON error."""
    if max_attempts <= 1:
        return model

    original = model.a_generate
    setattr(model, "_invalid_json_retries", 0)

    async def _retrying_a_generate(
        self: AmazonBedrockModel,
        prompt: str,
        schema: Optional[BaseModel] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        attempts_remaining = max_attempts

        while True:
            try:
                return await original(prompt, schema, *args, **kwargs)
            except ValueError as exc:
                message = str(exc).lower()
                if error_substring not in message or attempts_remaining <= 1:
                    raise

                attempts_remaining -= 1
                self._invalid_json_retries += 1  # type: ignore[attr-defined]
                logging.warning(
                    "Bedrock judge emitted invalid JSON; retrying (%s attempts left)",
                    attempts_remaining,
                )

    model.a_generate = MethodType(_retrying_a_generate, model)
    return model


def bedrock_retry_count(model: AmazonBedrockModel) -> int:
    """Return how many invalid-JSON retries this model has attempted."""
    return int(getattr(model, "_invalid_json_retries", 0))
