from botocore.exceptions import ClientError, NoCredentialsError

from govuk_chat_evaluation.aws_credentials import (
    AwsCredentialCheckResult,
    check_aws_credentials,
    ensure_aws_credentials,
)


def test_check_aws_credentials_success(mocker):
    mock_client = mocker.Mock()
    mock_client.get_caller_identity.return_value = {
        "Account": "123456789012",
        "Arn": "arn:aws:sts::123456789012:assumed-role/test-role/test-user",
        "UserId": "ABCDEF",
    }
    mock_session = mocker.Mock()
    mock_session.create_client.return_value = mock_client
    mocker.patch(
        "govuk_chat_evaluation.aws_credentials.Session", return_value=mock_session
    )

    result = check_aws_credentials(region="eu-west-1")

    assert result.ok is True
    assert result.account == "123456789012"
    assert result.arn is not None
    assert result.user_id == "ABCDEF"
    mock_session.create_client.assert_called_once_with("sts", region_name="eu-west-1")


def test_check_aws_credentials_missing(mocker):
    mock_client = mocker.Mock()
    mock_client.get_caller_identity.side_effect = NoCredentialsError()
    mock_session = mocker.Mock()
    mock_session.create_client.return_value = mock_client
    mocker.patch(
        "govuk_chat_evaluation.aws_credentials.Session", return_value=mock_session
    )

    result = check_aws_credentials()

    assert result.ok is False
    assert result.error is not None
    assert "credentials" in result.error.lower()


def test_check_aws_credentials_client_error(mocker):
    mock_client = mocker.Mock()
    mock_client.get_caller_identity.side_effect = ClientError(
        {
            "Error": {
                "Code": "ExpiredToken",
                "Message": "The security token included in the request is expired",
            }
        },
        "GetCallerIdentity",
    )
    mock_session = mocker.Mock()
    mock_session.create_client.return_value = mock_client
    mocker.patch(
        "govuk_chat_evaluation.aws_credentials.Session", return_value=mock_session
    )

    result = check_aws_credentials()

    assert result.ok is False
    assert result.error is not None
    assert "ExpiredToken" in result.error


def test_ensure_aws_credentials_rechecks_after_script(tmp_path, mocker):
    (tmp_path / ".env.aws").write_text("")

    check_mock = mocker.patch(
        "govuk_chat_evaluation.aws_credentials.check_aws_credentials",
        side_effect=[
            AwsCredentialCheckResult(ok=False, error="ExpiredToken"),
            AwsCredentialCheckResult(ok=True),
        ],
    )
    run_mock = mocker.patch(
        "govuk_chat_evaluation.aws_credentials.run_export_aws_credentials_script"
    )
    load_mock = mocker.patch("govuk_chat_evaluation.aws_credentials.load_env_aws")

    result = ensure_aws_credentials(tmp_path)

    assert result.ok is True
    run_mock.assert_called_once_with(tmp_path)
    assert check_mock.call_count == 2
    assert load_mock.call_count == 2
    load_mock.assert_any_call(tmp_path, override=False)
    load_mock.assert_any_call(tmp_path, override=True)


def test_ensure_aws_credentials_skips_script_when_already_valid(tmp_path, mocker):
    check_mock = mocker.patch(
        "govuk_chat_evaluation.aws_credentials.check_aws_credentials",
        return_value=AwsCredentialCheckResult(ok=True),
    )
    run_mock = mocker.patch(
        "govuk_chat_evaluation.aws_credentials.run_export_aws_credentials_script"
    )
    load_mock = mocker.patch("govuk_chat_evaluation.aws_credentials.load_env_aws")

    result = ensure_aws_credentials(tmp_path)

    assert result.ok is True
    check_mock.assert_called_once()
    run_mock.assert_not_called()
    load_mock.assert_called_once_with(tmp_path, override=False)
