#!/usr/bin/env bash
set -euo pipefail

aws_role="${1:-govuk-test-developer}"

echo "== Configuring AWS credentials for role: $aws_role =="

role_details=$(gds aws "$aws_role" --details 2>/dev/null) || {
  echo "ERROR: Couldn't establish AWS role: $aws_role" >&2
  exit 1
}

script_real_path="$(readlink -f "${BASH_SOURCE[0]}")"
repo_root="$(cd "$(dirname "$script_real_path")"/.. && pwd)"
config_path="${repo_root}/.env.aws"

has_expected_role() {
  local role_details="$1"
  local current_identity="$2"
  local expected_account expected_role actual_account actual_role

  expected_account=$(grep "AccountNumber" <<< "$role_details" | awk '{print $2}')
  expected_role=$(grep "RoleName" <<< "$role_details" | awk '{print $2}')

  if [[ $current_identity =~ arn:aws:sts::([0-9]+):assumed-role/([^/]+) ]]; then
    actual_account="${BASH_REMATCH[1]}"
    actual_role="${BASH_REMATCH[2]}"
  else
    echo "Couldn't match an assumed role in AWS identity: $current_identity" >&2
    echo "AWS output may have changed since this script was written. Exiting."
    exit 1
  fi

  [[ "$expected_account" == "$actual_account" && "$expected_role" == "$actual_role" ]]
}


if [[ -f "$config_path" ]]; then
  echo "Existing credentials found, checking validity"

  current_identity=$(uvx --from python-dotenv[cli] \
    dotenv -f "$config_path" run --override \
    -- \
    aws sts get-caller-identity 2>/dev/null || true)

  if [[ -z "$current_identity" ]]; then
    echo "Existing credentials are invalid or expired. Fetching new credentials..."
  elif ! has_expected_role "$role_details" "$current_identity"; then
    echo "Existing credentials are for a different role. Fetching new credentials..."
  else
    echo "Existing AWS credentials are still valid. No update needed."
    exit 0
  fi
else
  echo "No existing credentials file. Fetching new credentials..."
fi

aws_credentials="$(gds aws "$aws_role" -e --art 8h)"
relevant_credentials=$(echo "$aws_credentials" \
  | grep '^export ' \
  | sed -E 's/^export (.*);$/\1/')

printf '%s\n' "$relevant_credentials" > "$config_path"

echo "Written to $config_path"
