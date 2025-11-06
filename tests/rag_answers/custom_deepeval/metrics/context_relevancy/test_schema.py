from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.context_relevancy.schema import (
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

    class TestUnmetNeeds:
        def test_unmet_needs_with_no_verdicts(self):
            verdict1 = Verdict(verdict="yes", reason=None)
            verdict2 = Verdict(verdict="no", reason="Need more info")
            verdict3 = Verdict(verdict="no", reason="Clarify context")
            verdict_collection = VerdictCollection(
                verdicts=[verdict1, verdict2, verdict3]
            )

            unmet_needs = verdict_collection.unmet_needs()
            assert unmet_needs == ["Need more info", "Clarify context"]

        def test_unmet_needs_without_no_verdicts(self):
            verdict1 = Verdict(verdict="yes", reason=None)
            verdict2 = Verdict(verdict="idk", reason=None)
            verdict_collection = VerdictCollection(verdicts=[verdict1, verdict2])

            unmet_needs = verdict_collection.unmet_needs()
            assert unmet_needs == []
