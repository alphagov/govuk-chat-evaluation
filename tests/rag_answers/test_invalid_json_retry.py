from collections.abc import Sequence
from typing import Callable
from unittest.mock import AsyncMock

import logging
import pytest
from deepeval.models.base_model import DeepEvalBaseLLM

from govuk_chat_evaluation.rag_answers.invalid_json_retry import (
    attach_invalid_json_retry_to_model,
)


INVALID_JSON_ERROR = (
    "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
)


@pytest.fixture
def make_wrapped_model(mocker) -> Callable[..., tuple[DeepEvalBaseLLM, AsyncMock]]:
    def _make(outcomes: Sequence[object], *, max_attempts: int = 3):
        original = mocker.AsyncMock(side_effect=list(outcomes))
        model: DeepEvalBaseLLM = mocker.create_autospec(DeepEvalBaseLLM, instance=True)
        model.a_generate = original
        attach_invalid_json_retry_to_model(model, max_attempts=max_attempts)
        return model, original

    return _make


class TestAttachInvalidJsonRetryToModel:
    def test_attach_invalid_json_retry_to_model_requires_positive_attempts(
        self, mocker
    ):
        model: DeepEvalBaseLLM = mocker.create_autospec(DeepEvalBaseLLM, instance=True)
        model.a_generate = mocker.AsyncMock()

        with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
            attach_invalid_json_retry_to_model(model, max_attempts=0)

        with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
            attach_invalid_json_retry_to_model(model, max_attempts=-5)

    @pytest.mark.asyncio
    async def test_attach_invalid_json_retry_to_model_wraps_a_generate(self, mocker):
        original = mocker.AsyncMock(return_value=("ok", 0))
        model: DeepEvalBaseLLM = mocker.create_autospec(DeepEvalBaseLLM, instance=True)
        model.a_generate = original

        attach_invalid_json_retry_to_model(model, max_attempts=1)

        assert model.a_generate is not original
        result = await model.a_generate("prompt")
        assert result == ("ok", 0)
        original.assert_awaited_once()

    class TestMonkeyPatchedAGenerate:
        @pytest.mark.asyncio
        async def test_wrapped_a_generate_returns_without_retry(
            self, make_wrapped_model
        ):
            model, original = make_wrapped_model([("ok", 0)])

            result = await model.a_generate("prompt")

            assert result == ("ok", 0)
            original.assert_awaited_once()

        @pytest.mark.asyncio
        async def test_wrapped_a_generate_retries_after_invalid_json(
            self, make_wrapped_model
        ):
            model, original = make_wrapped_model(
                [ValueError(INVALID_JSON_ERROR), ("ok", 0)]
            )

            result = await model.a_generate("prompt")

            assert result == ("ok", 0)
            original.assert_awaited()
            assert original.await_count == 2

        @pytest.mark.asyncio
        async def test_wrapped_a_generate_propagates_unexpected_error(
            self, make_wrapped_model
        ):
            model, original = make_wrapped_model([ValueError("other error")])

            with pytest.raises(ValueError, match="other error"):
                await model.a_generate("prompt")

            original.assert_awaited_once()

        @pytest.mark.asyncio
        async def test_wrapped_a_generate_raises_after_exhausting_invalid_json(
            self, make_wrapped_model
        ):
            model, original = make_wrapped_model(
                [ValueError(INVALID_JSON_ERROR) for _ in range(3)]
            )

            with pytest.raises(ValueError, match="invalid JSON"):
                await model.a_generate("prompt")

            assert original.await_count == 3

        @pytest.mark.asyncio
        async def test_wrapped_a_generate_logs_attempts_remaining(
            self, make_wrapped_model, caplog
        ):
            caplog.set_level(logging.WARNING)
            model, _ = make_wrapped_model(
                [ValueError(INVALID_JSON_ERROR), ("ok", 0)],
                max_attempts=3,
            )

            await model.a_generate("prompt")

            assert "2 attempts left" in caplog.text

        @pytest.mark.asyncio
        async def test_wrapped_a_generate_logs_last_attempt(
            self, make_wrapped_model, caplog
        ):
            caplog.set_level(logging.WARNING)
            model, _ = make_wrapped_model(
                [ValueError(INVALID_JSON_ERROR), ("ok", 0)],
                max_attempts=2,
            )

            await model.a_generate("prompt")

            assert "last attempt remaining" in caplog.text
            assert "attempts left" not in caplog.text
