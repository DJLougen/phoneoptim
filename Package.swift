// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "phoneoptim",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "PhoneFastCore", targets: ["PhoneFastCore"]),
        .library(name: "PhoneFastModel", targets: ["PhoneFastModel"]),
        .library(name: "PhoneFastHarness", targets: ["PhoneFastHarness"]),
        // Per-track editable runners. benchmark.json assigns Runner/E2B to
        // track e2b and Runner/E4B to track e4b so their editablePaths do not
        // overlap.
        .library(name: "TrackRunnerE2B", targets: ["TrackRunnerE2B"]),
        .library(name: "TrackRunnerE4B", targets: ["TrackRunnerE4B"]),
        .executable(name: "bench-worker", targets: ["BenchWorker"]),
    ],
    dependencies: [
        // Vendored (not a submodule) so the Metal kernel sources are an
        // editable path. Nested Cmlx submodules (mlx, mlx-c) are copied in as
        // plain files. Root path dependency wins over mlx-swift-lm's remote
        // `from: 0.31.6` requirement for the same package identity; SwiftPM
        // reports the override as a conflicting-identity warning.
        .package(path: "Vendor/mlx-swift"),
        // Pinned upstream main (604fae7). The Gemma 4 VLM + assistant drafter
        // live here; trusted for now.
        // FoundationModelsIntegration default trait disabled: the
        // MLXFoundationModels target does not compile against the macOS 27
        // SDK and nothing here uses the adapter.
        .package(path: "Vendor/mlx-swift-lm", traits: []),
        // Tokenizers only (AutoTokenizer for local-directory loads).
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(name: "PhoneFastCore"),
        .target(
            name: "PhoneFastModel",
            dependencies: [
                "PhoneFastCore",
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "TrackRunnerE2B",
            dependencies: [
                "PhoneFastCore",
                "PhoneFastModel",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Runner/E2B"
        ),
        .target(
            name: "TrackRunnerE4B",
            dependencies: [
                "PhoneFastCore",
                "PhoneFastModel",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Runner/E4B"
        ),
        .target(
            name: "PhoneFastHarness",
            dependencies: [
                "PhoneFastCore",
                "PhoneFastModel",
                "TrackRunnerE2B",
                "TrackRunnerE4B",
            ]
        ),
        .executableTarget(
            name: "BenchWorker",
            dependencies: [
                "PhoneFastCore",
                "PhoneFastHarness",
                "PhoneFastModel",
                "TrackRunnerE2B",
                "TrackRunnerE4B",
            ],
            path: "Sources/BenchWorker"
        ),
        .testTarget(
            name: "PhoneFastTests",
            dependencies: [
                "PhoneFastCore",
                "PhoneFastHarness",
                "PhoneFastModel",
                "TrackRunnerE2B",
            ]
        ),
    ]
)
