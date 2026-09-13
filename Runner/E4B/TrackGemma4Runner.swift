import Foundation
import PhoneFastModel

/// Track `e4b` editable runner. Mirrors Runner/E2B; the E4B checkpoint (42
/// layers, hidden 2560) loads through the same `Gemma4Engine` path.
public final class TrackGemma4Runner {

    private let engine = Gemma4Engine()

    public init() {}

    public func load(weights: URL, assistant: URL?) async throws {
        try await engine.load(weights: weights, assistant: assistant)
    }

    public func run(
        prompt: String,
        image: URL?,
        maxTokens: Int,
        draftDepth: Int
    ) async throws -> Gemma4RunStats {
        try await engine.generate(
            prompt: prompt, image: image, maxTokens: maxTokens,
            draftDepth: draftDepth)
    }
}
