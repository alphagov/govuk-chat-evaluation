#!/usr/bin/env bash
set -euo pipefail

echo "== Configuring AWS credentials =="

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
config_path="${repo_root}/.env.aws"

aws_credentials="$(gds aws govuk-test-developer -e --art 8h)"
relevant_credentials=$(echo "$aws_credentials" \
  | grep '^export ' \
  | sed -E 's/^export (.*);$/\1/')

printf '%s\n' "$relevant_credentials" > "$config_path"

echo "Written to $(basename "$config_path")"
