#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/ssdwork/liuhaohan/petsgaussianhair}"
PACKAGE_PATH="${1:?usage: deploy_uploaded_clean_package.sh PACKAGE_PATH EXPECTED_SHA256}"
EXPECTED_SHA256="${2:?usage: deploy_uploaded_clean_package.sh PACKAGE_PATH EXPECTED_SHA256}"

case "$PROJECT" in
  /ssdwork/liuhaohan/petsgaussianhair) ;;
  *)
    echo "refusing unexpected PROJECT=$PROJECT" >&2
    exit 2
    ;;
esac

if [ ! -f "$PACKAGE_PATH" ]; then
  echo "package not found: $PACKAGE_PATH" >&2
  exit 3
fi

ACTUAL_SHA256="$(sha256sum "$PACKAGE_PATH")"
ACTUAL_SHA256="${ACTUAL_SHA256%% *}"
EXPECTED_SHA256="$(printf '%s' "$EXPECTED_SHA256" | tr '[:upper:]' '[:lower:]')"

if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "hash mismatch: expected=$EXPECTED_SHA256 actual=$ACTUAL_SHA256" >&2
  exit 4
fi

FORBIDDEN='(^|/)(__pycache__|_downloads|outputs|data_sources|external|refs|server_pull|\.git/|_stage1_check|.*\.pyc$)'
if tar -tzf "$PACKAGE_PATH" | grep -E "$FORBIDDEN"; then
  echo "package contains forbidden paths" >&2
  exit 5
fi

cd "$PROJECT"
if [ "$PWD" != "$PROJECT" ]; then
  echo "failed to enter project root: $PWD" >&2
  exit 6
fi

rm -rf anigroom configs docs scripts tools README.md .gitignore .gitattributes
tar -xzf "$PACKAGE_PATH"

if find anigroom configs scripts tools -maxdepth 3 -type d -name __pycache__ -print -quit | grep -q .; then
  echo "remote package extraction produced __pycache__" >&2
  exit 7
fi

printf 'deployed_package=%s\n' "$PACKAGE_PATH"
printf 'deployed_sha256=%s\n' "$ACTUAL_SHA256"
for path in anigroom configs docs scripts tools; do
  if [ -d "$path" ]; then
    find "$path" -maxdepth 1 -mindepth 0 -type d -print
  fi
done | sort
