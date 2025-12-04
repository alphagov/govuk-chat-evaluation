import pytest
from deepeval.test_case import LLMTestCase

from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.coherence import (
    CoherenceMetric,
)


@pytest.mark.real_bedrock
class TestCoherenceBedrockOpenAI:
    """
    Test the CoherenceMetric with real Bedrock API calls.
    This test requires the Bedrock environment variable to be set with
    `./scripts/export_aws_credentials.sh`.

    It can be run with the command:
    uv run pytest -m 'real_bedrock'
    """

    @pytest.mark.parametrize(
        "llm_test_case, expected_to_pass",
        [
            (
                LLMTestCase(
                    input="What is PIP?",
                    actual_output=(
                        "Personal Independence Payment (PIP) can help with extra living costs if "
                        "you have both a long-term physical or mental health condition or "
                        "disability and difficulty doing certain everyday tasks or getting around "
                        "because of your condition.\n\n"
                        "There are 2 parts to PIP:\n\n"
                        "* a daily living part - if you need help with everyday tasks\n"
                        "* a mobility part - if you need help with getting around\n\n"
                        "Whether you get one or both parts and how much you get depends on how "
                        "difficult you find everyday tasks and getting around.\n\n"
                        "You can get PIP even if you're working, have savings or are getting most "
                        "other benefits.\n\n"
                        "[Check if you're eligible for PIP][1] on GOV.UK.\n\n\n\n"
                        "[1]: https://www.gov.uk/pip/eligibility"
                    ),
                ),
                True,
            ),
            (
                LLMTestCase(
                    input="What is PIP?",
                    actual_output="I don't know the answer to that question.",
                ),
                True,
            ),
            (
                LLMTestCase(
                    input="How do I renew my passport before I was born?",
                    actual_output="I cannot answer that. Please try asking something else.",
                ),
                True,
            ),
            (
                LLMTestCase(
                    input=(
                        "I'm a tenant in England. Can my landlord raise my rent mid-tenancy?"
                    ),
                    actual_output=(
                        "If you're a tenant, you can increase the rent by giving notice to your tenant. "
                        "However, landlords must also ensure they follow the correct process before agreeing "
                        "to rent increases for themselves. Rent can go up and down at any time unless it's fixed, "
                        "and landlords must pay more if they want to increase their rent. If you disagree, you "
                        "should serve notice to yourself or contact a solicitor to challenge it."
                    ),
                ),
                False,
            ),
            (
                LLMTestCase(
                    input="Do I need to tell HMRC if I start self-employment?",
                    actual_output=(
                        "You must register for Self Assessment when you start self-employment. "
                        "You can do this online through your Government Gateway account. "
                        "You may need to register if your income is over a certain threshold. "
                        "Registration is optional for many people, but recommended."
                    ),
                ),
                False,
            ),
            (
                LLMTestCase(
                    input="How do I request flexible working?",
                    actual_output=(
                        "You can make a flexible working request after 26 weeks of employment. "
                        "Write to your employer explaining the changes you want. "
                        "As an employer, you must consider the request and respond within 2 months. "
                        "You should include when you want the changes to start and how they'll affect your work."
                    ),
                ),
                False,
            ),
        ],
        ids=[
            "coherent_comprehensive_answer_pip",
            "coherent_i_dont_know_pip",
            "coherent_refusal_nonsensical_input_passport",
            "incoherent_role_confusion_rent_increase",
            "incoherent_contradictory_obligations_self_assessment",
            "incoherent_perspective_slip_flexible_working",
        ],
    )
    @pytest.mark.asyncio
    async def test_coherence_score_bedrock(
        self,
        bedrock_openai_judge,
        llm_test_case: LLMTestCase,
        expected_to_pass: bool,
    ):
        metric = CoherenceMetric(
            model=bedrock_openai_judge,
            include_reason=False,
        )

        await metric.a_measure(llm_test_case)

        assert metric.is_successful() == expected_to_pass
