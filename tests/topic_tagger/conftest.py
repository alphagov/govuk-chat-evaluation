import json
import pytest


@pytest.fixture
def mock_data():
    return [
        {
            "question": "Question 1",
            "expected_primary_topic": "benefits",
            "actual_primary_topic": "benefits",
            "expected_secondary_topic": "tax",
            "actual_secondary_topic": "tax",
        },
        {
            "question": "Question 2",
            "expected_primary_topic": "benefits",
            "actual_primary_topic": "tax",
            "expected_secondary_topic": None,
            "actual_secondary_topic": None,
        },
        {
            "question": "Question 3",
            "expected_primary_topic": "tax",
            "actual_primary_topic": "benefits",
            "expected_secondary_topic": None,
            "actual_secondary_topic": "tax",
        },
    ]


@pytest.fixture
def mock_input_data(mock_project_root, mock_data):
    path = mock_project_root / "input_data.jsonl"

    with open(path, "w") as file:
        for item in mock_data:
            json.dump(item, file)
            file.write("\n")

    return path
