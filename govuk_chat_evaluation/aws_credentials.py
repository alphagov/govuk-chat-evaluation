from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess

from botocore.exceptions import (  # type: ignore
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)
from botocore.session import Session
from dotenv import load_dotenv


@dataclass(frozen=True)
class AwsCredentialCheckResult:
    ok: bool
    account: str | None = None
    arn: str | None = None
    user_id: str | None = None
    error: str | None = None


class AwsCredentialScriptError(RuntimeError):
    pass


def _resolve_region(region: str | None) -> str:
    if region:
        return region
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("AWS_BEDROCK_REGION")
        or "eu-west-1"
    )


def check_aws_credentials(region: str | None = None) -> AwsCredentialCheckResult:
    resolved_region = _resolve_region(region)
    session = Session()

    try:
        client = session.create_client("sts", region_name=resolved_region)
        response = client.get_caller_identity()
        return AwsCredentialCheckResult(
            ok=True,
            account=response.get("Account"),
            arn=response.get("Arn"),
            user_id=response.get("UserId"),
        )
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


def run_export_aws_credentials_script(project_root: Path) -> None:
    script_path = project_root / "scripts" / "export_aws_credentials.sh"
    command = ["bash", str(script_path)]

    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AwsCredentialScriptError(
            "Failed to run scripts/export_aws_credentials.sh. "
            "Check gds/aws tooling and try again."
        ) from exc


def load_env_aws(project_root: Path, override: bool) -> None:
    env_path = project_root / ".env.aws"
    if env_path.exists():
        load_dotenv(env_path, override=override)


def ensure_aws_credentials(project_root: Path) -> AwsCredentialCheckResult:
    load_env_aws(project_root, override=False)

    result = check_aws_credentials()
    if result.ok:
        return result

    run_export_aws_credentials_script(project_root)
    load_env_aws(project_root, override=True)
    return check_aws_credentials()
