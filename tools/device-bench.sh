#!/usr/bin/env bash
# Build, install, and run the on-device benchmark, then pull results back.
#
# Usage:
#   tools/device-bench.sh [spec-json]      # optional: bench-spec.json to push
#
# Env: DEVICE_ID, BUNDLE_ID, RESULTS_DIR (default /tmp/phoneOptim-device).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"

# Resolve the optional spec path before cd so a caller-relative path works.
SPEC_ARG=""
if (( "$#" > 0 )) && [[ -f "$1" ]]; then
  SPEC_ARG="$(cd "$(dirname "$1")" >/dev/null && pwd -P)/$(basename "$1")"
fi

cd "${ROOT_DIR}/BenchApp"

DEVICE_ID="${DEVICE_ID:-07DEF158-56A2-586F-B3A2-FA3BAEA4882C}"
BUNDLE_ID="${BUNDLE_ID:-com.djl.phoneoptim.bench}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/phoneOptim-device}"
SCHEME=PhoneOptimBench
PROJ=PhoneOptimBench.xcodeproj
DD=$(xcodebuild -project "${PROJ}" -scheme "${SCHEME}" -destination "id=${DEVICE_ID}" -showBuildSettings 2>/dev/null | awk '/ BUILD_DIR /{print $3; exit}')
echo "device-bench: BUILD_DIR=${DD}"

echo "device-bench: build-for-testing"
xcodebuild -project "${PROJ}" -scheme "${SCHEME}" \
  -destination "id=${DEVICE_ID}" \
  -allowProvisioningUpdates \
  build-for-testing 2>&1 | tail -3

APP="${DD}/Debug-iphoneos/PhoneOptimBench.app"
XCTESTRUN=$(find "$(dirname "${DD}")/Products" ~/Library/Developer/Xcode/DerivedData/PhoneOptimBench-*/Build/Products \
  -name '*.xctestrun' 2>/dev/null | head -1)
[[ -n "${XCTESTRUN}" ]] || { echo "device-bench: no .xctestrun found" >&2; exit 1; }
echo "device-bench: xctestrun=${XCTESTRUN}"

echo "device-bench: installing ${APP}"
xcrun devicectl device install app --device "${DEVICE_ID}" "${APP}"

if [[ -n "${SPEC_ARG}" ]]; then
  echo "device-bench: pushing spec ${SPEC_ARG}"
  xcrun devicectl device copy to --device "${DEVICE_ID}" \
    --domain-type appDataContainer --domain-identifier "${BUNDLE_ID}" \
    --source "${SPEC_ARG}" --destination "Documents/bench-spec.json"

  # A relative goldenPath in the spec resolves against Documents; push the
  # matching file from correctness_prompts/ when it exists locally.
  GOLDEN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("goldenPath") or "")' "${SPEC_ARG}" 2>/dev/null || true)
  if [[ -n "${GOLDEN}" && "${GOLDEN}" != /* && -f "${ROOT_DIR}/correctness_prompts/${GOLDEN}" ]]; then
    echo "device-bench: pushing golden correctness_prompts/${GOLDEN}"
    xcrun devicectl device copy to --device "${DEVICE_ID}" \
      --domain-type appDataContainer --domain-identifier "${BUNDLE_ID}" \
      --source "${ROOT_DIR}/correctness_prompts/${GOLDEN}" \
      --destination "Documents/${GOLDEN}"
  fi
fi

echo "device-bench: test-without-building"
xcodebuild test-without-building \
  -xctestrun "${XCTESTRUN}" \
  -destination "id=${DEVICE_ID}" 2>&1 | grep -E 'BENCH|Test Suite|passed|failed' | tail -25 || true

mkdir -p "${RESULTS_DIR}"
echo "device-bench: pulling Documents/results -> ${RESULTS_DIR}"
xcrun devicectl device copy from --device "${DEVICE_ID}" \
  --domain-type appDataContainer --domain-identifier "${BUNDLE_ID}" \
  --source "Documents/results" --destination "${RESULTS_DIR}/results" || true
ls -la "${RESULTS_DIR}/results" 2>/dev/null || true
