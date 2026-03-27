#!/usr/bin/env bash
#
# Generate checksum file for Elixir NIF binaries from GitHub release.
# Must run BEFORE mix compile — RustlerPrecompiled validates checksums at compile time.
#
# Usage: ./generate_elixir_checksums.sh <version>

set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
REPO="kreuzberg-dev/liter-llm"
CHECKSUM_FILE="packages/elixir/checksum-Elixir.LiterLlm.Native.exs"

TARGETS=(
  "aarch64-apple-darwin"
  "aarch64-unknown-linux-gnu"
  "x86_64-unknown-linux-gnu"
)

NIF_VERSIONS=("2.16" "2.17")

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Generating checksums for v${VERSION}..."

CHECKSUMS=()

for TARGET in "${TARGETS[@]}"; do
  for NIF_VERSION in "${NIF_VERSIONS[@]}"; do
    if [[ "$TARGET" == *"windows"* ]]; then
      EXT="dll"
    else
      EXT="so"
    fi

    FILENAME="libliter_llm_rustler-v${VERSION}-nif-${NIF_VERSION}-${TARGET}.${EXT}.tar.gz"
    URL="https://github.com/${REPO}/releases/download/v${VERSION}/${FILENAME}"

    echo "Downloading: $FILENAME"

    if curl -fsSL -o "${TMPDIR}/${FILENAME}" "$URL"; then
      if command -v sha256sum &>/dev/null; then
        CHECKSUM=$(sha256sum "${TMPDIR}/${FILENAME}" | cut -d' ' -f1)
      elif command -v shasum &>/dev/null; then
        CHECKSUM=$(shasum -a 256 "${TMPDIR}/${FILENAME}" | cut -d' ' -f1)
      else
        echo "ERROR: No sha256sum or shasum command found"
        exit 1
      fi

      echo "  Checksum: sha256:${CHECKSUM}"
      CHECKSUMS+=("  \"${FILENAME}\" => \"sha256:${CHECKSUM}\",")
    else
      echo "  WARNING: Failed to download $FILENAME (skipping)"
    fi
  done
done

if [ ${#CHECKSUMS[@]} -eq 0 ]; then
  echo "ERROR: No checksums generated"
  exit 1
fi

mapfile -t SORTED_CHECKSUMS < <(printf '%s\n' "${CHECKSUMS[@]}" | sort)

echo "Writing checksum file: $CHECKSUM_FILE"
{
  echo "%{"
  for CHECKSUM in "${SORTED_CHECKSUMS[@]}"; do
    echo "$CHECKSUM"
  done
  echo "}"
} >"$CHECKSUM_FILE"

echo "Done! Generated checksums for ${#SORTED_CHECKSUMS[@]} files."
cat "$CHECKSUM_FILE"
