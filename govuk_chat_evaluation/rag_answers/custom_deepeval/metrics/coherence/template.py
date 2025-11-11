RUBRIC = (
    # 1
    "Fundamentally incoherent for the user's practical needs. Contains contradictions about who decides/acts or what the user should do, makes it impossible to know what applies to them, or shifts between incompatible perspectives. A reasonable user would finish reading feeling confused about their situation or next steps. The answer may read smoothly but fails logically.",
    # 2
    "Significant coherence issues that create real risk of misinterpretation. May bury critical context (eligibility, conditions, 'it depends') until later in the answer, adopt the wrong perspective for extended portions, or present information in an order that requires re-reading to understand. Users could reasonably misunderstand their obligations, rights, or what actions to take.",
    # 3
    "Noticeable issues that disrupt practical understanding. Examples include: mixing 'you'll agree together' with 'they'll tell you' framing, important caveats appearing after definitive statements, ambiguity about whether something is required or optional, or unclear transitions between different scenarios. The answer is followable but requires careful reading to avoid misinterpretation.",
    # 4
    "Clear and well-structured with only minor inefficiencies. May have slight redundancy that doesn't confuse, could order information slightly better, or makes reasonable assumptions that might not apply to all users (but doesn't present them as universal truths). The core message is unambiguous and users can confidently act on it.",
    # 5
    "Unambiguously clear with no risk of misinterpretation. Critical context front-loaded, consistent perspective throughout, logical flow, free from contradiction, and impossible to misunderstand what applies to the user or what they should do. Maintains clarity even for complex scenarios.",
)
SCORE_RANGE = (1, len(RUBRIC))

COHERENCE_EVALUATION_TEMPLATE = """
You are evaluating the *coherence* of an AI-generated response from a UK government chatbot.

Coherence refers to how logically consistent, internally aligned, and well-structured the response is as a single, unified message.

Your task is to assess the coherence of the response by reasoning through its logic, structure, and flow, then provide:
1. A focused written explanation ("reason") describing your assessment.
2. A final numeric score ("score") between {min_score} and {max_score} based on the rubric.

Follow the evaluation steps and rubric below to assess the response.

Evaluation Steps:
STEP 1 - Establish User Context: Read the user's question carefully to identify: (a) their role/perspective (employee, tenant, parent, visa holder, etc.), (b) their specific situation or implied circumstances, (c) what they actually need to know or do. Write this down explicitly before proceeding.
STEP 2 - Critical Context Check (MOST IMPORTANT): Does the FIRST sentence address the most critical information for the user's decision-making? Ask: 'If the user only reads the first sentence, would they know whether this applies to them?' Critical context includes: eligibility conditions they may not meet, 'it depends on X', location/date limitations, or clear yes/no answers. If critical context appears later than the second sentence, this is a MAJOR flaw - consider scores 1-3.
STEP 3 - Contradiction Detection (requires careful reasoning): Read through the entire answer looking specifically for statements that contradict each other, even subtly. Pay special attention to: (a) Who makes decisions ('you decide' vs 'they decide' vs 'you agree together'), (b) What's required vs optional ('you must' vs 'you can' vs 'you may'), (c) Whether something applies ('this applies to you' vs 'this may apply'). Write down any contradictions found. Even ONE meaningful contradiction on core obligations/actions should result in score 1-2.
STEP 4 - Perspective Consistency Analysis: The entire answer must maintain the user's perspective from Step 1. Check: (a) Is 'you/your' always referring to the same person (the user)?, (b) If discussing other parties (employer, landlord, HMRC), is it always clear they are separate from 'you'?, (c) Does the answer ever slip into giving advice for the OTHER party's perspective? Write down the perspective used in each paragraph. Any perspective slip is a MAJOR issue - consider scores 1-3.
STEP 5 - Ambiguity Assessment: Identify any statements where a reasonable user could interpret the meaning in multiple ways. Look for: (a) Unclear pronouns ('they', 'this', 'it' - who/what?), (b) Vague timeframes ('soon', 'typically'), (c) Unclear scope ('you may need to' - is this required or not?), (d) Statements that could apply to multiple scenarios without clarification. Each significant ambiguity affecting what the user should DO is a major flaw.
STEP 6 - Logical Flow Verification: Check if information builds logically or if the user must mentally reorder it. Ask: 'Does each sentence/paragraph follow naturally from the previous one?' and 'Would moving any sentence earlier make the answer clearer?' If critical information appears AFTER less important details, or if the answer jumps between topics without transitions, consider scores 2-4 depending on severity.
STEP 7 - Practical Utility Test: After reading the answer, ask: 'Could the user confidently explain back: (a) what applies to their situation, (b) what they need to do next, (c) what's in their control vs others' control?' If you cannot answer all three clearly, the answer has failed coherence - consider scores 1-3.
STEP 8 - Apply the Harshness Principle: LLM-generated text often APPEARS coherent because it flows smoothly. You must look BEYOND surface readability. Ask: 'Even though this reads nicely, could a stressed/distracted/average-literacy user misunderstand what they should do?' If yes, do not give a 5. Remember: Score 1-2 is for answers that logically fail the user, not just answers that are grammatically poor. A well-written answer with subtle contradictions deserves a 1.
STEP 9 - Final Coherence-Only Check: Confirm your evaluation focuses solely on: structure, clarity, consistency, perspective, and logical presentation. You are NOT evaluating: correctness, completeness, relevance, or politeness. An answer can say 'I don't know' and score a 5 if it's clear about what it does/doesn't know. An answer can be comprehensive and score a 1 if it contradicts itself.
STEP 10 - Score Assignment with Justification: Based on Steps 1-9, assign a score using this logic: (a) ANY contradiction on core actions/obligations = 1 or 2, (b) Wrong/mixed perspective or buried critical context = 2 or 3, (c) Notable ambiguity or flow issues = 3 or 4, (d) Minor inefficiency only = 4, (e) No issues detected = 5. Provide specific evidence from the answer for your score. Default to the LOWER score if uncertain between two.

Rubric:
{rubric}

Test Case:
Input:
{user_input}

Actual Output:
{actual_output}

Return your evaluation strictly as a JSON object with the following fields:
- "reason": your reasoning and assessment of the response’s coherence.
- "score": a numeric value between {min_score} and {max_score}, assigned according to the rubric and your reasoning above.

Only return valid JSON. Do not include any other commentary or formatting.

---

JSON:
"""


class CoherenceTemplate:
    @staticmethod
    def evaluate(
        user_input: str,
        actual_output: str,
    ) -> str:
        min_score, max_score = SCORE_RANGE

        rubric_text = "\n".join(
            f"{score}: {criterion}"
            for score, criterion in enumerate(RUBRIC, start=min_score)
        )

        return COHERENCE_EVALUATION_TEMPLATE.format(
            min_score=min_score,
            max_score=max_score,
            rubric=rubric_text,
            user_input=user_input,
            actual_output=actual_output,
        )
