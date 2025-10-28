from govuk_chat_evaluation.rag_answers.shared_data_models import (
    StructuredContext,
)


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
