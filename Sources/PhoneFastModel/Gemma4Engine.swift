import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

/// Per-run statistics surfaced by ``Gemma4Engine/generate``.
public struct Gemma4RunStats: Sendable {
    public var prefillTokens: Int = 0
    public var prefillMilliseconds: Double = 0
    public var decodeTokens: Int = 0
    public var decodeMilliseconds: Double = 0
    public var decodeTokensPerSecond: Double = 0
    public var stopReason: String = ""
    /// MTP only: total tokens proposed by the drafter.
    public var proposedDraftTokens: Int? = nil
    /// MTP only: total draft tokens accepted by the target.
    public var acceptedDraftTokens: Int? = nil
    /// MTP only: tokens emitted per verification round, i.e.
    /// `1 + acceptanceRate * draftDepth`. Nil for serial runs.
    public var meanDraftLength: Double? = nil
    /// Non-nil when the MTP iterator fell back to serial mid-stream.
    public var passthroughReason: String? = nil
    public var text: String = ""

    public init() {}
}

/// Loads a pinned Gemma 4 E-series target (text + vision towers via MLXVLM's
/// `Gemma4`; the checkpoint's audio tower weights are skipped by upstream's
/// sanitize — `Gemma4.swift` drops `audio_tower` keys at load) plus the
/// paired `gemma4_assistant` MTP drafter, and runs text or text+image
/// generation, serial or MTP-speculative.
///
/// Trusted shared model code: per-track editable seams live in `Runner/E2B`
/// and `Runner/E4B`.
/// Result of a teacher-forced check: the model is fed the golden sequence
/// and must argmax to each next golden token. A drift tripwire — unlike
/// free-running text compare it localizes the first divergent position.
public struct GoldenCheckResult: Sendable {
    public var matched: Int = 0
    public var total: Int = 0
    /// Index into the expected-token span of the first argmax that missed.
    public var firstDivergence: Int? = nil

    public init() {}
}

public final class Gemma4Engine: @unchecked Sendable {

    private var context: ModelContext?
    private var drafter: (any MTPDrafterModel)?

    public init() {}

    /// Load the target checkpoint (and, when `assistant` is non-nil, the MTP
    /// drafter) from local directories produced by `setup.sh`.
    public func load(weights: URL, assistant: URL?) async throws {
        await Gemma4AssistantRegistration.register()
        let tokenizerLoader = #huggingFaceTokenizerLoader()
        context = try await loadGemma4ContextLowMemory(
            modelDirectory: weights, tokenizerLoader: tokenizerLoader)
        if let assistant {
            let drafterContext = try await MTPDrafterModelFactory.shared.load(
                from: assistant, using: tokenizerLoader)
            drafter = drafterContext.model
        }
    }

    /// One generation pass. When `drafter` is loaded and `draftDepth` > 0 the
    /// run is MTP-speculative with block size `draftDepth + 1`; otherwise it
    /// is serial. Returns stats plus the generated text.
    public func generate(
        prompt: String,
        image: URL?,
        maxTokens: Int,
        draftDepth: Int
    ) async throws -> Gemma4RunStats {
        guard let context else { throw Gemma4EngineError.notLoaded }

        var images: [UserInput.Image] = []
        if let image {
            images.append(.url(image))
        }
        let input = try await context.processor.prepare(
            input: UserInput(prompt: prompt, images: images))
        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)

        let stream: AsyncStream<Generation>
        if let drafter, draftDepth > 0 {
            stream = try MLXLMCommon.generate(
                input: input, parameters: parameters, context: context,
                mtpDrafter: drafter, blockSize: draftDepth + 1)
        } else {
            stream = try MLXLMCommon.generate(
                input: input, parameters: parameters, context: context)
        }

        var stats = Gemma4RunStats()
        for await event in stream {
            switch event {
            case .chunk(let text):
                stats.text += text
            case .info(let info):
                stats.prefillTokens = info.promptTokenCount
                stats.prefillMilliseconds = info.promptTime * 1000
                stats.decodeTokens = info.generationTokenCount
                stats.decodeMilliseconds = info.generateTime * 1000
                stats.decodeTokensPerSecond = info.tokensPerSecond
                stats.stopReason = "\(info.stopReason)"
                stats.proposedDraftTokens = info.proposedDraftTokens
                stats.acceptedDraftTokens = info.acceptedDraftTokens
                stats.passthroughReason = info.passthroughReason
                if let proposed = info.proposedDraftTokens,
                    let accepted = info.acceptedDraftTokens, proposed > 0
                {
                    let acceptance = Double(accepted) / Double(proposed)
                    stats.meanDraftLength = 1 + acceptance * Double(draftDepth)
                }
            case .toolCall, .rejectedToolCall:
                break
            }
        }
        return stats
    }

    /// Greedy generation returning raw token ids plus the prepared prompt
    /// ids (chat template applied). Used to mint golden files.
    public func generatedTokenIds(
        prompt: String,
        image: URL?,
        maxTokens: Int
    ) async throws -> (promptTokenIds: [Int], generated: [Int]) {
        guard let context else { throw Gemma4EngineError.notLoaded }
        var images: [UserInput.Image] = []
        if let image {
            images.append(.url(image))
        }
        let input = try await context.processor.prepare(
            input: UserInput(prompt: prompt, images: images))
        let promptIds = input.text.tokens.asArray(Int32.self).map(Int.init)
        var generated: [Int] = []
        let stream = try MLXLMCommon.generateTokens(
            input: input,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            context: context)
        for await event in stream {
            if case .token(let t) = event { generated.append(t) }
        }
        return (promptIds, generated)
    }

    /// One forward over `promptTokenIds + expectedTokenIds`; each position's
    /// argmax must equal the next golden token. `firstDivergence` is the
    /// index within `expectedTokenIds` where the model first disagrees.
    public func teacherForcedCheck(
        promptTokenIds: [Int],
        expectedTokenIds: [Int]
    ) -> GoldenCheckResult {
        var result = GoldenCheckResult()
        guard let context, !promptTokenIds.isEmpty, !expectedTokenIds.isEmpty else {
            return result
        }
        let all = promptTokenIds + expectedTokenIds
        let tokens = MLXArray(all.map(Int32.init))[.newAxis]
        let out = context.model(
            LMInput.Text(tokens: tokens), cache: nil, state: nil)
        // logits[0, i, :].argmax predicts token i+1; the expected span starts
        // at the last prompt position.
        let start = promptTokenIds.count - 1
        let preds = argMax(out.logits, axis: -1)[0, start..<(all.count - 1)]
        let expected = MLXArray(expectedTokenIds.map(Int32.init))
        let hits = (preds .== expected).asType(.int32).asArray(Int32.self)
        for (i, hit) in hits.enumerated() {
            if hit != 0 {
                result.matched += 1
            } else if result.firstDivergence == nil {
                result.firstDivergence = i
            }
        }
        result.total = hits.count
        return result
    }
}

public enum Gemma4EngineError: Error {
    case notLoaded
}
