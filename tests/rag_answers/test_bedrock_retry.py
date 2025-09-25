from types import SimpleNamespace
from typing import Sequence, cast
from unittest.mock import AsyncMock

import pytest
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel

from govuk_chat_evaluation.rag_answers.bedrock_retry import (
    attach_retry_to_bedrock_model,
    bedrock_retry_count,
)


INVALID_JSON_ERROR = (
    "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
)


def _make_model_with_outcomes(outcomes: Sequence[object], max_attempts: int = 3):
    """Create a stubbed Bedrock model whose a_generate consumes the provided outcomes."""
    a_generate = AsyncMock(side_effect=list(outcomes))
    model = cast(AmazonBedrockModel, SimpleNamespace(a_generate=a_generate))
    attach_retry_to_bedrock_model(model, max_attempts=max_attempts)
    return model, a_generate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes, expected_result, expected_calls, expected_retries",
    [
        ([("ok", 0)], ("ok", 0), 1, 0),
        ([ValueError(INVALID_JSON_ERROR), ("ok", 0)], ("ok", 0), 2, 1),
    ],
    ids=["no-retry", "retry-once"],
)
async def test_retry_succeeds_after_expected_attempts(
    outcomes, expected_result, expected_calls, expected_retries
):
    model, a_generate = _make_model_with_outcomes(outcomes)

    result = await model.a_generate("prompt")

    assert result == expected_result
    assert a_generate.await_count == expected_calls
    assert bedrock_retry_count(model) == expected_retries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes, expected_calls, expected_retries, expected_message",
    [
        ([ValueError(INVALID_JSON_ERROR) for _ in range(3)], 3, 2, "invalid JSON"),
        ([ValueError("some other error")], 1, 0, "some other error"),
    ],
    ids=["invalid-json-exhausted", "other-value-error"],
)
async def test_retry_raises_after_invalid_or_unexpected_errors(
    outcomes, expected_calls, expected_retries, expected_message
):
    model, a_generate = _make_model_with_outcomes(outcomes)

    with pytest.raises(ValueError, match=expected_message):
        await model.a_generate("prompt")

    assert a_generate.await_count == expected_calls
    assert bedrock_retry_count(model) == expected_retries
