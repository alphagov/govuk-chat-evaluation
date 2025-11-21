from collections import defaultdict
from typing import TypeVar
from .data_models import GenerateInput

InputType = TypeVar("InputType", bound=GenerateInput)


def ensure_model_ids_are_unique(inputs: list[InputType]) -> list[InputType]:
    ids = [input.id for input in inputs]

    if len(ids) == len(set(ids)):
        return inputs

    inputs_by_id = defaultdict(list)

    for input in inputs:
        inputs_by_id[input.id].append(input)

    for _id, grouped_inputs in inputs_by_id.items():
        if len(grouped_inputs) > 1:
            for index, input in enumerate(grouped_inputs[1:], start=1):
                input.id = f"{input.id}-duplicate-{index}"

    return inputs
