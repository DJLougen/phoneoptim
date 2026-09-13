# TASK

Make Gemma 4 E2B (track `e2b`) and E4B (track `e4b`) generate as fast as
possible on an iPhone 16 Pro Max (8 GB), without changing the model's
observable behavior.

## Constraints

- Target quantization is frozen as shipped (see `README.md` pins table).
- Edit only your track's `editablePaths` (`benchmark.json`).
- Dependencies are limited to the vendored `mlx-swift`, `mlx-swift-lm`, and
  `swift-transformers`.
- The audio tower is not yet supported by mlx-swift-lm — text and text+image
  workloads only.

## Measure

`bench-worker` prints one JSON object per run. Measure serial and MTP
(draft depth 1–3) on the same machine before and after a change. Local
numbers are directional, not ranked.

## Pinned checkpoints

`setup.sh` fetches and sha256-verifies both tracks' targets and assistants;
revisions are pinned in `benchmark.json` and `Sources/PhoneFastCore`.
