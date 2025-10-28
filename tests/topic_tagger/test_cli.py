import pytest
import yaml
from click.testing import CliRunner
from datetime import datetime

from govuk_chat_evaluation.topic_tagger.cli import main
from govuk_chat_evaluation.topic_tagger.evaluate import EvaluationResult

FROZEN_TIME = datetime.now().replace(microsecond=0)

@pytest.fixture(autouse=True)
def mock_config_file(tmp_path, mock_input_data):
    """Write a config file as an input for testing"""
    data = {
        "what": "Testing Topic Tagger evaluations",
        "provider": "claude",
        "generate": True,
        "input_path": str(mock_input_data),
    }
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as file:
        yaml.dump(data, file)

    yield str(file_path)


@pytest.fixture(autouse=True)
def freeze_time_for_all_tests(freezer):
    """Automatically freeze time for all tests in this file."""
    freezer.move_to(FROZEN_TIME)


@pytest.fixture
def mock_data_generation(mocker):
    """Mock the async generation function to return EvaluationResult instances"""
    return_value = [
        EvaluationResult(
            question="How do I start a business and what tax do I need to pay?",
            expected_primary_topic="business",
            actual_primary_topic="business",
            expected_secondary_topic="tax",
            actual_secondary_topic="tax",
        ),
        EvaluationResult(
            question="What benefits am I eligible for?",
            expected_primary_topic="benefits",
            actual_primary_topic="business",
            expected_secondary_topic=None,
            actual_secondary_topic=None,
        ),
    ]

    return mocker.patch(
        "govuk_chat_evaluation.topic_tagger.generate.generate_inputs_to_evaluation_results",
        return_value=return_value,
    )


@pytest.fixture
def mock_output_directory(mock_project_root):
    return mock_project_root / "results" / "topic_tagger" / FROZEN_TIME.isoformat()


@pytest.mark.usefixtures("mock_data_generation")
def test_main_creates_output_files(mock_output_directory, mock_config_file):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file])

    config_file = mock_output_directory / "config.yaml"
    results_file = mock_output_directory / "results.csv"
    aggregate_file = mock_output_directory / "aggregate.csv"

    assert result.exit_code == 0, result.output
    assert mock_output_directory.exists()
    assert results_file.exists()
    assert aggregate_file.exists()
    assert config_file.exists()


def test_main_generates_results(
    mock_output_directory, mock_config_file, mock_data_generation
):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--generate"])

    generated_file = mock_output_directory / "generated.jsonl"

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_called_once()
    assert generated_file.exists()


@pytest.mark.usefixtures("mock_output_directory")
def test_main_doesnt_generate_results(mock_config_file, mock_data_generation):
    runner = CliRunner()
    result = runner.invoke(main, [mock_config_file, "--no-generate"])

    assert result.exit_code == 0, result.output
    mock_data_generation.assert_not_called()
