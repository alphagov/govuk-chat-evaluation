from .data_models import GenerateInput
from collections import Counter
from typing import Sequence


def ensure_unique_model_ids(inputs: Sequence[GenerateInput]):
    ids = [input.id for input in inputs]
    duplicates = [id for id, count in Counter(ids).items() if count > 1]

    if duplicates:
        duplicate_ids = ", ".join(duplicates)
        raise ValueError(f"Duplicate IDs found in inputs: {duplicate_ids}")
