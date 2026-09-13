# phoneOptim — Gemma 4 E-series on iPhone 16 Pro Max

Yukon-style speedup benchmark: run **Gemma 4 E2B / E4B** (fully multimodal)
as fast as possible on an iPhone 16 Pro Max (A18 Pro, 8 GB) using MLX Swift.
This repository currently carries the scaffold and the macOS baseline; the
iOS bench app lands in a later track update.

## Tracks (schemaVersion 2)

| Track | Target | MTP assistant |
|---|---|---|
| `e2b` | `mlx-community/gemma-4-E2B-it-qat-4bit` @ `42f62737` | `mlx-community/gemma-4-E2B-it-qat-assistant-4bit` @ `7acb8d02` |
| `e4b` | `mlx-community/gemma-4-E4B-it-qat-4bit` @ `0f35c6f6` | `mlx-community/gemma-4-E4B-it-qat-assistant-4bit` @ `dac8da1a` |

Editable paths are non-overlapping per track: `Runner/E2B` vs `Runner/E4B`.
Track `e2b` additionally lists the vendored `Vendor/mlx-swift` Metal kernel
sources as editable. **TODO:** the `e4b` kernel editable-path list is
deliberately empty for now — port the same list when the track opens.

Shared model loading/generation lives in `Sources/PhoneFastModel` (trusted);
constants and the frozen quantization pins in `Sources/PhoneFastCore`.

## The model (E2B)

35-layer text decoder, hidden 1536, 8 query heads / 1 KV head, head_dim 256
(global_head_dim 512), 4:1 `sliding_attention`:`full_attention` hybrid with
sliding window 512, the last 20 layers share K/V (`num_kv_shared_layers`),
PLE per-layer input embeddings, double-wide MLP, tied embeddings, vocab
262144. The checkpoint also carries a **vision tower** (loaded; image
prompts work) and an **audio tower** — the audio encoder is NOT yet
implemented in mlx-swift-lm and its weights are skipped at load
(`Gemma4.swift` sanitize drops `audio_tower` keys). **TODO:** audio input.

The MTP drafter is a 4-layer `gemma4_assistant` model that consumes the
target's last hidden state and per-layer-type shared K/V; each round drafts
`blockSize - 1` tokens verified in one target pass.

## Pinned quantization

Derived from the checkpoints' own `config.json` `quantization` blocks —
see `Sources/PhoneFastCore/Gemma4TargetQuantizationPins.swift`:

| Field | E2B | E4B |
|---|---|---|
| Fallback | 4-bit affine, group 64 | same |
| Overrides | `mlp.{gate,down,up}_proj` → 8-bit, group 64, every layer (105 keys, 35 layers) | same families (126 keys, 42 layers) |
| Packed codes | `uint32` words | same |
| scales/biases | `bfloat16` | same |

The target quantization is frozen as shipped: do not re-quantize or
re-represent target weights.

## Memory budget (iPhone 16 Pro Max, 8 GB)

E2B target is ~4.3 GB on disk at QAT 4-bit; the assistant adds ~0.25 GB.
That leaves roughly 3 GB of headroom for KV cache, the vision tower's
activations, MLX allocator cache, and the OS. Expect to cap the MLX cache
and prefer sliding-window-friendly KV settings on device.

## Setup

```bash
./setup.sh                    # e2b: download + verify + release build
PHONEOPTIM_TRACK=e4b ./setup.sh   # e4b
PHONEOPTIM_VERIFY_ONLY=1 ./setup.sh   # re-verify staged bytes
```

`setup.sh` verifies every file against `fixtures/*.sha256` and builds
`.build/release/bench-worker`.

## Baseline run (macOS)

```bash
# serial
.build/release/bench-worker --weights reference_weights/gemma-4-E2B-it-qat-4bit \
  --prompt correctness_prompts/short.txt --max-tokens 64 --serial

# MTP speculative, draft depth 3
.build/release/bench-worker --weights reference_weights/gemma-4-E2B-it-qat-4bit \
  --assistant reference_weights/gemma-4-E2B-it-qat-assistant-4bit \
  --prompt correctness_prompts/short.txt --max-tokens 64 --draft-depth 3

# text + image
.build/release/bench-worker --weights reference_weights/gemma-4-E2B-it-qat-4bit \
  --assistant reference_weights/gemma-4-E2B-it-qat-assistant-4bit \
  --prompt <file> --image <png> --max-tokens 64 --draft-depth 3
```

Output is a single JSON object: `prefill_tokens`, `prefill_ms`,
`decode_tokens`, `decode_ms`, `tps`, `mean_draft_len`, `output`.

## Layout

| Path | Role | Trust |
|---|---|---|
| `Runner/E2B`, `Runner/E4B` | Per-track editable runner seam | Editable (per track) |
| `Vendor/mlx-swift/` | Vendored MLX core; Metal kernels editable (e2b) | Mixed |
| `Vendor/mlx-swift-lm/` | Upstream mlx-swift-lm submodule @ `604fae7` | Trusted |
| `Sources/PhoneFastModel` | Shared Gemma 4 engine wiring | Trusted |
| `Sources/PhoneFastCore` | Constants, quantization pins | Trusted |
| `Sources/BenchWorker` | macOS benchmark CLI | Trusted |
| `fixtures/` | sha256 manifests for pinned checkpoints | Trusted |
