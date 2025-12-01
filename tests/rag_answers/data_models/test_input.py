import pytest
from deepeval.test_case import LLMTestCase
from govuk_chat_evaluation.rag_answers.data_models import (
    EvaluationTestCase,
    StructuredContext,
    GenerateInput,
)
import uuid
from unittest.mock import patch


class TestStructuredContext:
    def test_to_flattened_string(self):
        structured_context = StructuredContext(
            title="VAT",
            heading_hierarchy=["Tax", "VAT"],
            description="VAT overview",
            html_content="<p>Some HTML about VAT</p>",
            exact_path="https://gov.uk/vat",
            base_path="https://gov.uk",
        )

        flattened_string = structured_context.to_flattened_string()
        expected_string = "VAT\nTax > VAT\nVAT overview\n\n<p>Some HTML about VAT</p>"
        assert flattened_string == expected_string

    def test_to_flattened_context_content(self):
        structured_context = StructuredContext(
            title="VAT",
            heading_hierarchy=["Tax", "VAT"],
            description="VAT overview",
            html_content="<p>Some HTML about VAT</p>",
            exact_path="https://gov.uk/vat",
            base_path="https://gov.uk",
        )

        flattened_content = structured_context.to_flattened_context_content()
        expected_content = (
            "Context:\nPage Title: VAT\nPage description: VAT overview\n"
            "Headings: Tax > VAT\n\nContent:\n<p>Some HTML about VAT</p>"
        )
        assert flattened_content == expected_content


class TestEvaluationTestCase:
    @pytest.mark.parametrize("ideal_answer", ["Great", None])
    def test_to_llm_test_case(self, ideal_answer):
        """Test EvaluationTestCase.to_llm_test_case with and without ideal_answer"""
        structured_context = StructuredContext(
            title="VAT",
            heading_hierarchy=["Tax", "VAT"],
            description="VAT overview",
            html_content="<p>Some HTML about VAT</p>",
            exact_path="https://gov.uk/vat",
            base_path="https://gov.uk",
        )

        evaluation_test_case = EvaluationTestCase(
            question="How are you?",
            ideal_answer=ideal_answer,
            llm_answer="Fine",
            structured_contexts=[structured_context],
        )

        llm_test_case = evaluation_test_case.to_llm_test_case()

        assert isinstance(llm_test_case, LLMTestCase)
        assert isinstance(llm_test_case.name, str)
        assert llm_test_case.expected_output == ideal_answer
        assert llm_test_case.actual_output == evaluation_test_case.llm_answer

        assert isinstance(llm_test_case.retrieval_context, list)
        assert all(isinstance(chunk, str) for chunk in llm_test_case.retrieval_context)
        assert "VAT" in llm_test_case.retrieval_context[0]
        assert "Some HTML about VAT" in llm_test_case.retrieval_context[0]


class TestGenerateInput:
    def test_generate_input_id_defaults_to_uuid(self):
        expected_uuid = str(uuid.uuid4())
        with patch("uuid.uuid4", return_value=(expected_uuid)):
            input = GenerateInput(
                question="What is VAT?",
                ideal_answer="Value Added Tax",
            )

            assert input.id == expected_uuid
