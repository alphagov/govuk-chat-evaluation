from dataclasses import dataclass

from botocore.exceptions import (  # type: ignore
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)
from botocore.session import Session


@dataclass(frozen=True)
class AwsCredentialCheckResult:
    ok: bool
    error: str | None = None


def check_aws_credentials(*, region: str) -> AwsCredentialCheckResult:
    session = Session()

    try:
        client = session.create_client("sts", region_name=region)
        client.get_caller_identity()
        return AwsCredentialCheckResult(ok=True)
    except (NoCredentialsError, PartialCredentialsError) as exc:
        return AwsCredentialCheckResult(
            ok=False, error=str(exc) or "Missing AWS credentials"
        )
    except ClientError as exc:
        error = exc.response.get("Error", {})
        message = error.get("Message") or str(exc)
        code = error.get("Code") or "ClientError"
        return AwsCredentialCheckResult(ok=False, error=f"{code}: {message}")
    except Exception as exc:  # pragma: no cover - defensive for unexpected failures
        return AwsCredentialCheckResult(ok=False, error=str(exc))
