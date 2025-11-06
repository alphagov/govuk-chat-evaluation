class ContextRelevancyTemplate:
    @staticmethod
    def information_needs(input: str):
        return f"""You are given a user query directed to a UK government chatbot about content they expect to be reasonably available on GOV.UK.
            The user expects an answer that provides the key factual information or official details implied by their question.

            Your task: Extract the KEY INFORMATION NEEDS from the query.

            Each information need should represent an essential piece of government information that a good GOV.UK answer would provide in order to satisfy the query.

            The goal is to identify the *crucial pieces of information* that would allow the user to progress to the **next meaningful step** in their information journey.
            Do not include follow-up steps, secondary details, or additional actions that might come later. Focus only on the key facts or requirements the user needs to understand or decide what to do next.

            Guidelines:
            - Focus only on the central, key information needs — do not include secondary or tangential details.
            - Keep the scope within the kind of factual or procedural information GOV.UK would reasonably provide. Do not include personal or lifestyle advice.
            - Limit outputs to the **crucial elements** needed to move the user forward — not every possible step in a longer process.
            - For queries that ask for “help finding”, “show me”, or “GOV.UK guidance on …”, reduce the need to the *main topic* and the *type of information* sought (e.g., “rules for …”, “criteria to …”, “how to apply for …”).
            - Extract only what is explicitly or reasonably implied by the query itself. Do not add background knowledge or assumptions.
            - Break the query into distinct, answerable components only if the question clearly implies multiple aspects of official guidance.
            - Output must be valid JSON only, with the key `"information_needs"` mapped to a list of strings.
            - Do not include any explanation, commentary, or extra fields outside the JSON.

            Examples:

            1. Query:
            "Can I get financial help for my heating bills?"

            JSON:
            {{
                "information_needs": [
                "The government schemes available to help with heating or energy bills.",
                "Eligibility criteria for receiving heating bill support."
                ]
            }}

            2. Query:
            "How do I register as a landlord?"

            JSON:
            {{
                "information_needs": [
                "The legal requirements and process for registering as a landlord."
                ]
            }}

            3. Query:
            "What support is available for childcare costs while studying at university?"

            JSON:
            {{
                "information_needs": [
                "The types of financial support available for childcare costs.",
                "Eligibility criteria for receiving childcare support while studying at university.",
                "How to apply for childcare support as a university student."
                ]
            }}

            ===== END OF EXAMPLES =====

            Query:
            {input}

            JSON:

    """

    @staticmethod
    def truths(retrieval_context: list):
        return f"""You will be given a list of strings.
            Each string has two elements:
            1. "Context" – metadata such as the page title, description and headings.
            2. "Content" – the actual text from that page.

            Your task: From each string, extract a list of factual statements ("facts") that can be inferred from the content, while retaining the contextual meaning provided by "Context".

            Guidelines:
            - Summarize the metadata in 'context'
            - Each fact must remain coherent and grounded in both the "Context" and the "Content".
            - Ensure the 'context' reflects the contextual meaning of the original string.
            - If the context limits the scope (e.g. "benefits in England"), ensure the facts reflect that scope.
            - Do not merge facts across different strings. Treat each string independently.
            - Do not add knowledge that is not present.
            - Output must be valid JSON only.

            Example:
            Input:

            {{
                [
                "Context:
                Page Title: Benefits
                Description: Eligibility for benefits in the UK
                Headings: Benefits > If you live in England",

                Content:
                You are eligible to apply for benefits if you are over 18 and a resident.
                You need to show proof of residency.",

                "Context:
                Page Title: Accessing the NHS
                Description: How to access NHS treatments
                Headings: Accessing the NHS > If you are visiting",

                Content:
                Visitors from outside the UK may have to pay for NHS treatment."

                ]
            }}


            Expected JSON output format:
            {{
                "truths": [
                    {{
                    "context": "Benefits you can access if you live in England",
                    "facts": [
                    "In England, you are eligible to apply for benefits if you are over 18.",
                    "In England, you are eligible to apply for benefits if you are a resident.",
                    "In England, to access benefits you need to show proof of residency."
                    ]
                }},
                {{
                    "context": "Healthcare entitlements for visitors",
                    "facts": [
                    "Visitors from outside the UK may have to pay for NHS treatment."
                    ]
                }}

            }}

            ===== END OF EXAMPLE =====

            Input:
            {retrieval_context}

            Output JSON:
        """

    @staticmethod
    def verdicts(information_needs: list[str], extracted_truths: list[dict]):
        return f"""You will be given:
            1. A list of INFORMATION NEEDS (each describing a specific question or factual requirement from the user query).
            2. A structured list of TRUTHS extracted from the retrieval context.
            Each item in the TRUTHS list includes:
                - "context": metadata such as a page title or description (e.g. "Benefits in England")
                - "facts": a list of factual statements relevant to that context.

            Your task:
            For each information need, decide whether that need is **met by information in the retrieval context**.
            “Met” means that the retrieval context provides a clear and relevant answer to that need — **even if the answer itself is negative or says something is not possible.**

            Each information need should only be considered met if it is addressed by facts from a **relevant context**.
            Facts from a context that clearly does not apply (e.g. "Benefits in England" when the information need concerns "Scotland") should **not** be used as evidence of coverage.
            However, if the context is general (e.g. "UK-wide"), it may apply to any relevant region unless the information need specifies otherwise.

            For each information need, output a JSON object with two fields:
            - "verdict": one of "yes", "no", or "idk"
            - "reason": a brief explanation of why you chose that verdict, referring to relevant contextual facts when applicable.

            **Definitions:**
            - Use `"yes"` if one or more facts from a matching or relevant context clearly answer or satisfy the information need (even if the answer is “no,” “not eligible,” or expressed differently but equivalent in meaning).
            - Use `"no"` if the retrieval context does not contain any relevant facts addressing that information need, or if all available facts come from an irrelevant context.
            - Use `"idk"` if the information need or the contexts are too vague, conflicting, or incomplete to make a confident judgment.

            **Important:**
            The verdict labels do **not** reflect the truth value of the answer. They only indicate whether the information need is covered by the retrieval context.

            Expected JSON format:
            {{
                "verdicts": [
                    {{
                        "verdict": "yes",
                        "reason": "<explanation_for_why_the_need_is_fully_addressed_by_relevant_facts_in_the_context>"
                    }},
                    {{
                        "verdict": "no",
                        "reason": "<explanation_for_why_the_need_is_not_addressed_or_the_context_is_irrelevant>"
                    }},
                    {{
                        "verdict": "idk",
                        "reason": "<explanation_for_uncertainty_or_partial_relevance>"
                    }}
                ]
            }}

            The number of verdicts MUST equal the number of information needs.

            Retrieval Context (structured by context):
            {extracted_truths}

            Information Needs:
            {information_needs}

            JSON:

            """

    @staticmethod
    def reason(score: float, input: str, unmet_needs: list[str]):
        return f"""Below is a list of un-met information needs. It is a list of strings explaining why the information needs in the 'query' were not met by the information present in the 'retrieval context'.
Given the context relevancy score, which is a 0-1 score indicating how well the information needs posed by the `query` were met by the retrieval context (higher the better), CONCISELY summarize the un-met needs to justify the score.

Expected JSON format:
{{
    "reason": "The score is <context_relevancy_score> because <your_reason>."
}}

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.

If there are no un-met information needs, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
Your reason MUST use information in `unmet_needs` in your reason.
Be sure in your reason, as if you know what the information need in the query is from the unmet needs.
**

Context Relevancy Score:
{score}

Query:
{input}

Unmet Needs:
{unmet_needs}

JSON:
"""
