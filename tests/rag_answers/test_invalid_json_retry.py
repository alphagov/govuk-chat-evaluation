from collections.abc import Sequence
from typing import Callable
from unittest.mock import AsyncMock

import pytest
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel

from govuk_chat_evaluation.rag_answers.invalid_json_retry import (
    attach_invalid_json_retry_to_model,
)


INVALID_JSON_ERROR = (
    "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
)


@pytest.fixture
def make_wrapped_model(mocker) -> Callable[..., tuple[AmazonBedrockModel, AsyncMock]]:
    def _make(outcomes: Sequence[object], *, max_attempts: int = 3):
        original = mocker.AsyncMock(side_effect=list(outcomes))
        model = mocker.Mock(spec=AmazonBedrockModel)
        model.a_generate = original
        attach_invalid_json_retry_to_model(model, max_attempts=max_attempts)
        return model, original

    return _make


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "outcomes",
        "expected_result",
        "expected_error_match",
        "expected_calls",
    ),
    [
        ([("ok", 0)], ("ok", 0), None, 1),
        ([ValueError(INVALID_JSON_ERROR), ("ok", 0)], ("ok", 0), None, 2),
        (
            [
                ValueError(INVALID_JSON_ERROR),
                ValueError(INVALID_JSON_ERROR),
                ("ok", 0),
            ],
            ("ok", 0),
            None,
            3,
        ),
        ([ValueError(INVALID_JSON_ERROR) for _ in range(3)], None, "invalid JSON", 3),
        ([ValueError("other error")], None, "other error", 1),
    ],
    ids=[
        "succeeds-immediately",
        "recovers-after-one-retry",
        "recovers-after-two-retries",
        "exhausts-invalid-json",
        "passes-through-other-error",
    ],
)
async def test_retry_flows(
    make_wrapped_model,
    outcomes,
    expected_result,
    expected_error_match,
    expected_calls,
):
    model, original = make_wrapped_model(outcomes)

    assert model.a_generate is not original

    if expected_error_match:
        with pytest.raises(ValueError, match=expected_error_match):
            await model.a_generate("prompt")
    else:
        result = await model.a_generate("prompt")
        assert result == expected_result

    original.assert_awaited()
    assert original.await_count == expected_calls
