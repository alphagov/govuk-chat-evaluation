from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.absence_of_factual_contradictions.schema import (
    Verdict,
    VerdictCollection,
)


class TestVerdictCollection:
    class TestScoreVerdicts:
        def test_score_verdicts_with_mixed_verdicts(self):
            verdict1 = Verdict(verdict="yes")
            verdict2 = Verdict(verdict="no")
            verdict3 = Verdict(verdict="idk")
            verdict_collection = VerdictCollection(
                verdicts=[verdict1, verdict2, verdict3]
            )

            score = verdict_collection.score_verdicts()
            assert score == 2 / 3

        def test_score_verdicts_all_no(self):
            verdict1 = Verdict(verdict="no")
            verdict2 = Verdict(verdict="no")
            verdict_collection = VerdictCollection(verdicts=[verdict1, verdict2])

            score = verdict_collection.score_verdicts()
            assert score == 0.0

        def test_score_verdicts_empty(self):
            verdict_collection = VerdictCollection(verdicts=[])

            score = verdict_collection.score_verdicts()
            assert score == 1.0

    class TestContradictionReasons:
        def test_contradiction_reasons_with_no_verdicts(self):
            verdict1 = Verdict(verdict="yes", reason=None)
            verdict1 = Verdict(verdict="yes", reason="Different departments")
            verdict2 = Verdict(verdict="no", reason="Conflicting assertions")
            verdict3 = Verdict(verdict="no", reason="Different dates")
            verdict_collection = VerdictCollection(
                verdicts=[verdict1, verdict2, verdict3]
            )

            contradiction_reasons = verdict_collection.contradiction_reasons()
            assert contradiction_reasons == [
                "Conflicting assertions",
                "Different dates",
            ]

        def test_contradiction_reasons_without_no_verdicts(self):
            verdict1 = Verdict(verdict="yes", reason=None)
            verdict2 = Verdict(verdict="idk", reason=None)
            verdict_collection = VerdictCollection(verdicts=[verdict1, verdict2])

            contradiction_reasons = verdict_collection.contradiction_reasons()
            assert contradiction_reasons == []
