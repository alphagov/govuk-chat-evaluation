from botocore.exceptions import ClientError, NoCredentialsError

from govuk_chat_evaluation.aws_credentials import (
    check_aws_credentials,
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
    assert result.error is None
    mock_session.create_client.assert_called_once_with("sts", region_name="eu-west-1")


def test_check_aws_credentials_missing(mocker):
    mock_client = mocker.Mock()
    mock_client.get_caller_identity.side_effect = NoCredentialsError()
    mock_session = mocker.Mock()
    mock_session.create_client.return_value = mock_client
    mocker.patch(
        "govuk_chat_evaluation.aws_credentials.Session", return_value=mock_session
    )

    result = check_aws_credentials(region="eu-west-1")

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

    result = check_aws_credentials(region="eu-west-1")

    assert result.ok is False
    assert result.error is not None
    assert "ExpiredToken" in result.error
