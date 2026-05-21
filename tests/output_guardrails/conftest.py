import json

import pytest


@pytest.fixture
def mock_input_data(mock_project_root):
    data = [
        {
            "message": "This answer contains inappropriate content.",
            "expected_triggered": True,
            "actual_triggered": True,
            "expected_guardrails": {"appropriate_language": True, "political": True},
            "actual_guardrails": {"appropriate_language": True, "political": True},
            "model": "model_name",
        },
        {
            "message": "This is a safe and appropriate answer.",
            "expected_triggered": False,
            "actual_triggered": True,
            "expected_guardrails": {"appropriate_language": False, "political": False},
            "actual_guardrails": {"appropriate_language": True, "political": True},
            "model": "model_name",
        },
    ]

    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    return path
