#!/usr/bin/env bash
# Push a pinned checkpoint into the PhoneOptimBench app's Documents on the
# device, via devicectl's appDataContainer domain.
#
# Usage:
#   tools/device-push-weights.sh [weights-dir [dest-name]]
#   tools/device-push-weights.sh            # pushes E2B target + assistant
#
# Env: DEVICE_ID (default: the paired iPhone), BUNDLE_ID.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-07DEF158-56A2-586F-B3A2-FA3BAEA4882C}"
BUNDLE_ID="${BUNDLE_ID:-com.djl.phoneoptim.bench}"

push() {
  local src="$1" dest="$2"
  echo "push-weights: ${src} -> Documents/${dest}"
  xcrun devicectl device copy to \
    --device "${DEVICE_ID}" \
    --domain-type appDataContainer \
    --domain-identifier "${BUNDLE_ID}" \
    --source "${src}" \
    --destination "Documents/${dest}"
}

if (( "$#" > 0 )); then
  push "$1" "${2:-$(basename "$1")}"
else
  push reference_weights/gemma-4-E2B-it-qat-4bit gemma-4-E2B-it-qat-4bit
  push reference_weights/gemma-4-E2B-it-qat-assistant-4bit \
    gemma-4-E2B-it-qat-assistant-4bit
fi
echo "push-weights: done"
