#!/usr/bin/env bash
# measure-and-score.sh -- benchmark.json's benchmarkCommand / preSubmitCommand
# for phoneoptim-gemma4-eseries. Runs the paired protocol (serial control vs
# MTP candidate) through bench-worker --protocol and converts the BenchResult
# JSON into the {score, metrics} shape Yukon's scorePath consumer expects:
# `score` a finite number, `metrics` an optional JSON object.
#
# Refusals (exit 1, no score.json written):
#   - missing weights / golden / worker (preflight failures)
#   - degenerateOutput: empty or all-<pad> generation
#   - outputsMatch false: serial and candidate texts diverge
#   - goldenMatchRate < 0.95: teacher-forced drift past the tripwire
#
# Usage:
#   tools/measure-and-score.sh --track e2b|e4b [--preflight-only]
#
# Env:
#   PHONEOPTIM_SCORE_PATH   score output path (default: score.json)
#   PHONEOPTIM_PAIRS        serial/candidate leg pairs (default: 2)
#   PHONEOPTIM_MAX_TOKENS   generated tokens per leg (default: 64)
#   PHONEOPTIM_DRAFT_DEPTH  candidate MTP draft depth (default: 3)
#   PHONEOPTIM_COOL_SECONDS GPU idle gate before timed legs (default: 5)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

TRACK=""
PREFLIGHT_ONLY=0
while (( "$#" > 0 )); do
  case "$1" in
    --track) TRACK="$2"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    *) echo "measure-and-score: unrecognized argument: $1" >&2; exit 2 ;;
  esac
done

case "${TRACK}" in
  e2b|e4b) ;;
  *) echo "measure-and-score: --track e2b|e4b is required" >&2; exit 2 ;;
esac

TRACK_UPPER="$(echo "${TRACK}" | tr 'a-z' 'A-Z')"
TARGET_DIR="reference_weights/gemma-4-${TRACK_UPPER}-it-qat-4bit"
ASSISTANT_DIR="reference_weights/gemma-4-${TRACK_UPPER}-it-qat-assistant-4bit"
GOLDEN="correctness_prompts/golden-${TRACK}-sky.json"
PROMPT="correctness_prompts/short.txt"
WORKER=".build/release/bench-worker"
SCORE_PATH="${PHONEOPTIM_SCORE_PATH:-score.json}"
RESULT_JSON="benchmark-result-${TRACK}.json"

PAIRS="${PHONEOPTIM_PAIRS:-2}"
MAX_TOKENS="${PHONEOPTIM_MAX_TOKENS:-64}"
DRAFT_DEPTH="${PHONEOPTIM_DRAFT_DEPTH:-3}"
COOL_SECONDS="${PHONEOPTIM_COOL_SECONDS:-5}"

fail() { echo "measure-and-score: $1" >&2; exit 1; }

[[ -x "${WORKER}" ]] || fail "bench-worker not built (${WORKER}); run ./setup.sh"
[[ -d "${TARGET_DIR}" ]] || fail "weights missing: ${TARGET_DIR}; run ./setup.sh"
[[ -d "${ASSISTANT_DIR}" ]] || fail "assistant missing: ${ASSISTANT_DIR}; run ./setup.sh"
[[ -f "${PROMPT}" ]] || fail "prompt missing: ${PROMPT}"
[[ -f "${GOLDEN}" ]] || fail "golden missing: ${GOLDEN}"

if (( PREFLIGHT_ONLY )); then
  echo "measure-and-score: preflight ok (track ${TRACK})"
  exit 0
fi

echo "measure-and-score: running paired protocol (track ${TRACK}, depth ${DRAFT_DEPTH}, pairs ${PAIRS})"
"${WORKER}" --protocol --track "${TRACK}" \
  --weights "${TARGET_DIR}" \
  --assistant "${ASSISTANT_DIR}" \
  --prompt "${PROMPT}" \
  --max-tokens "${MAX_TOKENS}" \
  --draft-depth "${DRAFT_DEPTH}" \
  --pairs "${PAIRS}" \
  --golden "${GOLDEN}" \
  --cool-seconds "${COOL_SECONDS}" \
  > "${RESULT_JSON}"

python3 - "${RESULT_JSON}" "${SCORE_PATH}" "${TRACK}" <<'PY'
import json, math, sys

result_path, score_path, track = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(result_path))

def refuse(why):
    print(f"measure-and-score: refusing to score: {why}", file=sys.stderr)
    sys.exit(1)

if r.get("degenerateOutput"):
    refuse("degenerateOutput is true (empty or all-<pad> generation)")
if not r.get("outputsMatch"):
    refuse("outputsMatch is false (serial and candidate texts diverge)")
rate = r.get("goldenMatchRate")
if rate is None:
    refuse("no goldenMatchRate (teacher-forced check did not run)")
if rate < 0.95:
    refuse(f"goldenMatchRate {rate:.3f} below 0.95 tripwire "
           f"(first divergence at position {r.get('goldenFirstDivergence')})")

score = r.get("composite")
if not isinstance(score, (int, float)) or not math.isfinite(score):
    refuse(f"composite is not a finite number: {score!r}")

legs = r.get("legs", [])
metrics = {
    "track": track,
    "device": r.get("device"),
    "prefillGain": r.get("prefillGain"),
    "decodeGain": r.get("decodeGain"),
    "composite": score,
    "serialDecodeMsPerToken": r.get("serialDecodeMsPerToken"),
    "candidateDecodeMsPerToken": r.get("candidateDecodeMsPerToken"),
    "serialPrefillMsPerToken": r.get("serialPrefillMsPerToken"),
    "candidatePrefillMsPerToken": r.get("candidatePrefillMsPerToken"),
    "goldenMatchRate": rate,
    "goldenFirstDivergence": r.get("goldenFirstDivergence"),
    "coolGateMs": r.get("coolGateMs"),
    "warmupMs": r.get("warmupMs"),
    "draftDepth": (r.get("spec") or {}).get("draftDepth"),
    "pairs": (r.get("spec") or {}).get("pairs"),
    "proposedDraftTokens": sum(l.get("proposedDraftTokens") or 0 for l in legs),
    "acceptedDraftTokens": sum(l.get("acceptedDraftTokens") or 0 for l in legs),
    "peakPhysFootprintBytes": max((l.get("physFootprintBytes") or 0 for l in legs), default=0),
    "legs": legs,
}

with open(score_path, "w") as f:
    json.dump({"score": score, "metrics": metrics}, f, indent=2, sort_keys=True)
    f.write("\n")

print(f"measure-and-score: score={score:.4f} "
      f"(prefillGain={metrics['prefillGain']:.4f} decodeGain={metrics['decodeGain']:.4f}) "
      f"-> {score_path}")
PY
