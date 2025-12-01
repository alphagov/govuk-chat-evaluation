import pytest
from pydantic import BaseModel
from unittest.mock import MagicMock, AsyncMock
from govuk_chat_evaluation.rag_answers.patch_bedrock_a_generate import (
    a_generate_filters_non_text_responses,
)
from botocore.exceptions import EndpointConnectionError
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel


@pytest.fixture
def model():
    model = AmazonBedrockModel.__new__(AmazonBedrockModel)
    model.model_id = "bedrock-model"
    model.input_token_cost = 0
    model.output_token_cost = 0
    model.generation_kwargs = {}

    return model


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def attach_client(model, mock_client):
    async def _ensure_client(self):
        return mock_client

    async def _close(self):
        pass

    model._ensure_client = _ensure_client.__get__(model)
    model.close = _close.__get__(model)

    return model


@pytest.fixture
def patched_model(attach_client):
    model = attach_client
    model.a_generate = a_generate_filters_non_text_responses.__get__(model)
    return model


@pytest.mark.asyncio
async def test_a_generate_filters_non_text_responses_calls_the_converse_api_correctly(
    patched_model, mock_client
):
    mock_client.converse = AsyncMock(
        return_value={
            "output": {
                "message": {
                    "content": [{"text": "Text content returned in first index."}]
                }
            },
            "usage": {"inputTokens": 5, "outputTokens": 7},
        }
    )

    await patched_model.a_generate("question")

    mock_client.converse.assert_called_once_with(
        modelId="bedrock-model",
        messages=[{"role": "user", "content": [{"text": "question"}]}],
        inferenceConfig={},
    )


@pytest.mark.asyncio
async def test_a_generate_filters_non_text_responses_with_single_text_message(
    patched_model, mock_client
):
    mock_client.converse = AsyncMock(
        return_value={
            "output": {
                "message": {
                    "content": [{"text": "Text content returned in first index."}]
                }
            },
            "usage": {"inputTokens": 5, "outputTokens": 7},
        }
    )

    text, cost = await patched_model.a_generate("question")

    assert text == "Text content returned in first index."
    assert cost == patched_model.calculate_cost(5, 7)


@pytest.mark.asyncio
async def test_a_generate_filters_non_text_responses_with_reasoning_message(
    patched_model, mock_client
):
    mock_client.converse = AsyncMock(
        return_value={
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "thinking..."}
                            }
                        },
                        {"text": "Text content returned in second index."},
                    ]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 20},
        }
    )

    text, cost = await patched_model.a_generate("question")

    assert text == "Text content returned in second index."
    assert cost == patched_model.calculate_cost(10, 20)


@pytest.mark.asyncio
async def test_a_generate_filters_non_text_responses_schema_parses_json(
    patched_model, mock_client
):
    class DummySchema(BaseModel):
        answer: str

    mock_client.converse = AsyncMock(
        return_value={
            "output": {
                "message": {
                    "content": [
                        {"text": '{"answer": "Text content returned in first index."}'}
                    ]
                }
            },
            "usage": {"inputTokens": 2, "outputTokens": 3},
        }
    )

    schema, cost = await patched_model.a_generate("question", schema=DummySchema)

    assert isinstance(schema, DummySchema)
    assert schema.answer == "Text content returned in first index."
    assert cost == patched_model.calculate_cost(2, 3)


@pytest.mark.asyncio
async def test_a_generate_retries_on_failure(patched_model, mock_client):
    mock_client.converse = AsyncMock(
        side_effect=[
            EndpointConnectionError(endpoint_url="https://bedrock.fake"),
            {
                "output": {
                    "message": {
                        "content": [
                            {"text": "Text content returned on the second attempt."}
                        ]
                    }
                },
                "usage": {"inputTokens": 1, "outputTokens": 1},
            },
        ]
    )

    text, cost = await patched_model.a_generate("question")

    assert text == "Text content returned on the second attempt."
    assert mock_client.converse.call_count == 2
