from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_precision_recall import (
    FactClassificationCache,
)
from govuk_chat_evaluation.rag_answers.custom_deepeval.metrics.factual_precision_recall.schema import (
    ClassifiedFacts,
)


class TestFactClassificationCache:
    def test_cache_get_and_set(self):
        cache = FactClassificationCache()
        classified_facts = ClassifiedFacts(TP=["t"], FP=[], FN=[])

        assert cache.get("model", "answer", "ground") is None

        cache.set("model", "answer", "ground", classified_facts)

        assert cache.get("model", "answer", "ground") == classified_facts

    def test_get_and_set_distinguish_keys_by_model_and_texts(self):
        cache = FactClassificationCache()

        facts_model_a = ClassifiedFacts(TP=["a"], FP=[], FN=[])
        facts_model_b = ClassifiedFacts(TP=["b"], FP=[], FN=[])
        facts_other_answer = ClassifiedFacts(TP=["c"], FP=[], FN=[])

        # Different models with same texts should not collide.
        cache.set("model-a", "answer", "ground", facts_model_a)
        cache.set("model-b", "answer", "ground", facts_model_b)

        assert cache.get("model-a", "answer", "ground") == facts_model_a
        assert cache.get("model-b", "answer", "ground") == facts_model_b

        # Different answers under the same model should not collide.
        cache.set("model-a", "different", "ground", facts_other_answer)

        assert cache.get("model-a", "answer", "ground") == facts_model_a
        assert cache.get("model-a", "different", "ground") == facts_other_answer
