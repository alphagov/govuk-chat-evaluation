#!/usr/bin/env bash
set -euo pipefail

aws_role="${1:-govuk-test-developer}"

echo "== Configuring AWS credentials for role: $aws_role =="

script_real_path="$(readlink -f "${BASH_SOURCE[0]}")"
repo_root="$(cd "$(dirname "$script_real_path")"/.. && pwd)"
config_path="${repo_root}/.env.aws"

credentials_valid() {
  local config_path="$1"
  (
    set -euo pipefail
    uvx --from python-dotenv[cli] \
        dotenv -f $config_path run --override \
        -- \
        aws sts get-caller-identity >/dev/null 2>&1
  )
}

if [[ -f "$config_path" ]]; then
  echo "Existing credentials found, checking validity"
  if credentials_valid "$config_path"; then
    echo "Existing AWS credentials are still valid. No update needed."
    exit 0
  else
    echo "Existing credentials are invalid or expired. Fetching new credentials..."
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
