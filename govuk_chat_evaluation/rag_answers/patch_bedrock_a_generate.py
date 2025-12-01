from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.constants import ProviderSlug as PS

retry_bedrock = create_retry_decorator(PS.BEDROCK)


@retry_bedrock
async def a_generate_filters_non_text_responses(self, prompt, schema=None):
    """
    Patched version of a_generate that safely extracts 'text'
    from bedrock hosted OpenAI OSS models. These return reasoningContent messages
    in the first index of the content messages list, which the original a_generate
    method does not handle correctly. It expects the first index to always contain
    a text response and attempts to parse it directly. This causes a KeyError to be
    raised.

    A PR has been submitted to deepeval to include this fix upstream.
    See: https://github.com/confident-ai/deepeval/pull/2328

    The original a_generate method can be found here:
    https://github.com/confident-ai/deepeval/blob/main/deepeval/models/llms/amazon_bedrock_model.py#L69
    """
    try:
        payload = self.get_converse_request_body(prompt)
        client = await self._ensure_client()

        response = await client.converse(
            modelId=self.model_id,
            messages=payload["messages"],
            inferenceConfig=payload["inferenceConfig"],
        )

        message_text = next(
            (
                item["text"]
                for item in response["output"]["message"]["content"]
                if "text" in item
            )
        )

        cost = self.calculate_cost(
            response["usage"]["inputTokens"],
            response["usage"]["outputTokens"],
        )

        if schema is None:
            return message_text, cost
        else:
            json_output = trim_and_load_json(message_text)
            return schema.model_validate(json_output), cost

    finally:
        await self.close()
