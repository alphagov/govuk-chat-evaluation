from deepeval.test_case import LLMTestCase
from pydantic import BaseModel
from typing import Optional
import uuid


class StructuredContext(BaseModel):
    title: str
    heading_hierarchy: list[str]
    description: Optional[str] = None
    html_content: str
    exact_path: str
    base_path: str

    def to_flattened_string(self) -> str:
        """Return the flattened string representation of the structure context."""
        return (
            f"{self.title}\n"
            f"{' > '.join(self.heading_hierarchy)}\n"
            f"{self.description}\n\n"
            f"{self.html_content}"
        )

    def to_flattened_context_content(self) -> str:
        """Return the flattened string representation of the structured chunk in two parts: context and content,
        with labels to describe each field"""
        return (
            f"Context:\n"
            f"Page Title: {self.title}\n"
            f"Page description: {self.description}\n"
            f"Headings: {' > '.join(self.heading_hierarchy)}\n\n"
            f"Content:\n"
            f"{self.html_content}"
        )


class GenerateInput(BaseModel):
    question: str
    ideal_answer: Optional[str] = None


class EvaluationTestCase(GenerateInput):
    llm_answer: str
    structured_contexts: list[StructuredContext]

    def to_llm_test_case(self) -> LLMTestCase:
        return LLMTestCase(
            input=self.question,
            name=str(uuid.uuid4()),
            expected_output=self.ideal_answer,
            actual_output=self.llm_answer,
            retrieval_context=[
                ctx.to_flattened_string() for ctx in self.structured_contexts
            ],
            additional_metadata={"structured_contexts": self.structured_contexts},
        )
