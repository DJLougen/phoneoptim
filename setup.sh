#!/usr/bin/env bash
# Provision the pinned Gemma 4 E2B target + MTP assistant into
# reference_weights/ and build the release bench worker.
#
# Mirrors the mlxfast tracks' contract: both checkpoints are pinned by
# full revision and every downloaded byte is verified against the checked-in
# sha256/size manifests in fixtures/. Anonymous public fetch via `hf`.
#
# Track e4b is pinned but not yet downloaded by default (the iOS bench app
# lands later); set PHONEOPTIM_TRACK=e4b to provision it instead.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK="${PHONEOPTIM_TRACK:-e2b}"
HF="${HF_BIN:-${HOME}/.local/bin/hf}"

case "${TRACK}" in
  e2b)
    TARGET_REPO="mlx-community/gemma-4-E2B-it-qat-4bit"
    TARGET_REV="42f62737af7a9fd8c1d55d79666c1a217be4e2e2"
    TARGET_MANIFEST="fixtures/reference_gemma4_e2b_qat4bit.sha256"
    ASSISTANT_REPO="mlx-community/gemma-4-E2B-it-qat-assistant-4bit"
    ASSISTANT_REV="7acb8d02364777be103dbf8c8dcb007f55f717e6"
    ASSISTANT_MANIFEST="fixtures/gemma4_e2b_assistant.sha256"
    ;;
  e4b)
    TARGET_REPO="mlx-community/gemma-4-E4B-it-qat-4bit"
    TARGET_REV="0f35c6f6d386f7f74e628bd7c6526ce531212300"
    TARGET_MANIFEST="fixtures/reference_gemma4_e4b_qat4bit.sha256"
    ASSISTANT_REPO="mlx-community/gemma-4-E4B-it-qat-assistant-4bit"
    ASSISTANT_REV="dac8da1a066d5e00bc085e281fa73f2d5ec2fbe0"
    ASSISTANT_MANIFEST="fixtures/gemma4_e4b_assistant.sha256"
    ;;
  *)
    echo "setup.sh: unknown PHONEOPTIM_TRACK '${TRACK}' (e2b|e4b)" >&2
    exit 2
    ;;
esac

TARGET_DIR="reference_weights/${TARGET_REPO##*/}"
ASSISTANT_DIR="reference_weights/${ASSISTANT_REPO##*/}"

verify_manifest() {
  local manifest="$1" dir="$2"
  echo "setup.sh: verifying ${dir} against ${manifest}"
  local rc=0
  while read -r sha size path; do
    case "${sha}" in ''|'#'*) continue ;; esac
    local f="${dir}/${path}"
    if [[ ! -f "${f}" ]]; then
      echo "setup.sh: MISSING ${f}" >&2; rc=1; continue
    fi
    local actual_size actual_sha
    actual_size=$(stat -f%z "${f}")
    if [[ "${actual_size}" != "${size}" ]]; then
      echo "setup.sh: SIZE MISMATCH ${path}: ${actual_size} != ${size}" >&2; rc=1; continue
    fi
    actual_sha=$(shasum -a 256 "${f}" | awk '{print $1}')
    if [[ "${actual_sha}" != "${sha}" ]]; then
      echo "setup.sh: SHA256 MISMATCH ${path}" >&2; rc=1
    fi
  done < "${manifest}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "setup.sh: verification failed for ${dir}" >&2
    exit 1
  fi
  echo "setup.sh: ${dir} verified"
}

download() {
  local repo="$1" rev="$2" dir="$3" manifest="$4"
  if [[ ! -x "${HF}" ]]; then
    HF="$(command -v hf || true)"
  fi
  [[ -n "${HF}" ]] || { echo "setup.sh: hf CLI not found" >&2; exit 1; }
  if [[ "${PHONEOPTIM_VERIFY_ONLY:-0}" == "1" ]]; then
    verify_manifest "${manifest}" "${dir}"
    return
  fi
  echo "setup.sh: downloading ${repo} @ ${rev} -> ${dir}"
  "${HF}" download "${repo}" --revision "${rev}" --local-dir "${dir}"
  verify_manifest "${manifest}" "${dir}"
}

download "${TARGET_REPO}" "${TARGET_REV}" "${TARGET_DIR}" "${TARGET_MANIFEST}"
download "${ASSISTANT_REPO}" "${ASSISTANT_REV}" "${ASSISTANT_DIR}" "${ASSISTANT_MANIFEST}"

if [[ "${PHONEOPTIM_SKIP_SWIFT_BUILD:-0}" != "1" ]]; then
  echo "setup.sh: building bench-worker (release)"
  swift build -c release --product bench-worker
fi

echo "setup.sh: done (track ${TRACK})"
