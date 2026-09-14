import Foundation
import PhoneFastModel

/// Track `e2b` editable runner. Owns the per-track seam between the trusted
/// `PhoneFastModel` engine and the bench worker: this is where track
/// participants optimize the Gemma 4 E2B load/generate path (prompt
/// preparation, cache configuration, draft-depth policy, kernel dispatch).
///
/// Baseline: straight wiring over `Gemma4Engine` — no caching tricks, greedy
/// decoding, block size = draftDepth + 1.
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

    public func generatedTokenIds(
        prompt: String,
        image: URL?,
        maxTokens: Int
    ) async throws -> (promptTokenIds: [Int], generated: [Int]) {
        try await engine.generatedTokenIds(
            prompt: prompt, image: image, maxTokens: maxTokens)
    }

    public func teacherForcedCheck(
        promptTokenIds: [Int],
        expectedTokenIds: [Int]
    ) -> GoldenCheckResult {
        engine.teacherForcedCheck(
            promptTokenIds: promptTokenIds, expectedTokenIds: expectedTokenIds)
    }
}
