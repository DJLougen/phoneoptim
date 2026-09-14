import XCTest

import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Tokenizers
import PhoneFastCore
import PhoneFastHarness
import PhoneFastModel
@testable import PhoneOptimBench

/// Headless driver: `xcodebuild test -destination 'id=<device>'` runs this
/// inside the host app so Documents is the app container. The result JSON is
/// written to Documents/results/ AND attached to the test.
final class BenchRunnerTests: XCTestCase {

    func testRunBenchProtocol() async throws {
        let result = try await BenchProtocol.run(spec: BenchRunner.loadSpec()) { line in
            BenchRunner.logProgress(line)
            NSLog("BENCH-PROGRESS %@", line)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let resultsDir = docs.appendingPathComponent("results", isDirectory: true)
        try FileManager.default.createDirectory(at: resultsDir, withIntermediateDirectories: true)
        let name =
            "bench-\(result.spec.track)-d\(result.spec.draftDepth)-\(Int(Date().timeIntervalSince1970)).json"
        let outURL = resultsDir.appendingPathComponent(name)
        try data.write(to: outURL)

        add(XCTAttachment(contentsOfFile: outURL))
        NSLog("BENCH-RESULT \(outURL.path)")
        NSLog(
            "BENCH composite=%.3f serialDecode=%.1ftok/s candDecode=%.1ftok/s match=%@",
            result.composite,
            result.serialDecodeMsPerToken > 0 ? 1000 / result.serialDecodeMsPerToken : 0,
            result.candidateDecodeMsPerToken > 0 ? 1000 / result.candidateDecodeMsPerToken : 0,
            result.outputsMatch ? "yes" : "no")
        for leg in result.legs {
            NSLog(
                "BENCH-LEG %@ pair=%d prefill=%.1fms/%dtok decode=%.0fms/%dtok mem=%lluMB thermal=%d->%d",
                leg.role, leg.pairIndex, leg.prefillMs, leg.prefillTokens,
                leg.decodeMs, leg.decodeTokens, leg.physFootprintBytes / 1_048_576,
                leg.thermalStateBefore, leg.thermalStateAfter)
        }
        if let rate = result.goldenMatchRate {
            NSLog(
                "BENCH golden rate=%.3f firstDivergence=%@",
                rate,
                result.goldenFirstDivergence.map(String.init) ?? "none")
            // Minted on macOS, verified on iOS Metal kernels: near-tie argmax
            // flips are possible, so the gate is a drift tripwire not a
            // bitwise contract.
            XCTAssertGreaterThanOrEqual(rate, 0.95)
        }
        XCTAssertFalse(result.degenerateOutput)
        XCTAssertTrue(result.outputsMatch)
    }

    /// Exercises the disk-resident PLE embedding: gathers a few rows through
    /// the module (pread + GPU dequantize) and checks for sane, non-NaN
    /// values. Reference ranges come from the macOS/python dequantize of the
    /// same rows (row 5: min -0.223, max 0.207; row 0: min -0.801, max 0.863).
    func testPLEProbe() async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let weightsURL = docs.appendingPathComponent("gemma-4-E2B-it-qat-4bit")

        let ctx = try await loadGemma4ContextLowMemory(
            modelDirectory: weightsURL, tokenizerLoader: #huggingFaceTokenizerLoader())

        guard let ple = ctx.model.leafModules().flattened().first(where: {
            $0.0.hasSuffix("embed_tokens_per_layer")
        })?.1 as? QuantizedEmbedding else {
            XCTFail("no embed_tokens_per_layer module")
            return
        }

        let out = ple(MLXArray([Int32(0), Int32(5), Int32(12345)]))
        let nan = isNaN(out).asType(.int32).sum().item(Int.self)
        NSLog(
            "PLE-PROBE out shape=%@ dtype=%@ nan=%d",
            "\(out.shape)", "\(out.dtype)", nan)
        NSLog(
            "PLE-PROBE row0 min=%.3f max=%.3f row5 min=%.3f max=%.3f row12345 min=%.3f max=%.3f",
            out[0].min().item(Float.self), out[0].max().item(Float.self),
            out[1].min().item(Float.self), out[1].max().item(Float.self),
            out[2].min().item(Float.self), out[2].max().item(Float.self))
        XCTAssertEqual(nan, 0)
        XCTAssertEqual(out[1].min().item(Float.self), -0.223, accuracy: 0.002)
        XCTAssertEqual(out[1].max().item(Float.self), 0.207, accuracy: 0.002)
    }

    /// Logs the input ids for the benchmark prompt and the first generated
    /// token ids, so we can tell apart "tokenizer emitted pad ids" from
    /// "model forward produced pad logits" from "detokenizer is broken".
    func testTokenProbe() async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let weightsURL = docs.appendingPathComponent("gemma-4-E2B-it-qat-4bit")

        let ctx = try await loadGemma4ContextLowMemory(
            modelDirectory: weightsURL, tokenizerLoader: #huggingFaceTokenizerLoader())

        let input = try await ctx.processor.prepare(
            input: UserInput(prompt: "Explain why the sky is blue in two sentences."))
        let ids = input.text.tokens.asArray(Int32.self)
        NSLog("TOKEN-PROBE inputIds=%@", ids.map(String.init).joined(separator: ","))

        // The bench calls generate() (Generation events). Try it too — if it
        // pads while generateTokens is clean, the difference is in the layer.
        var genText = ""
        let gstream = try MLXLMCommon.generate(
            input: input,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            context: ctx)
        for await event in gstream {
            if case .chunk(let t) = event { genText += t }
        }
        NSLog("TOKEN-PROBE generate-text=%@", genText)

        var generated: [Int] = []
        let stream = try MLXLMCommon.generateTokens(
            input: input,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            context: ctx)
        for await event in stream {
            if case .token(let t) = event { generated.append(t) }
        }
        NSLog("TOKEN-PROBE generated=%@", generated.map(String.init).joined(separator: ","))
        NSLog(
            "TOKEN-PROBE decoded=%@",
            (try? ctx.tokenizer.decode(tokenIds: generated)) ?? "<decode failed>")

        // One direct forward: check the last-position logits for NaN and see
        // which token argmax picks.
        let out = ctx.model(input.text, cache: nil, state: nil)
        let last = out.logits[0, -1, 0...]
        let nanCount = isNaN(last).asType(.int32).sum().item(Int.self)
        let maxV = last.max().item(Float.self)
        let minV = last.min().item(Float.self)
        let am = argMax(last).item(Int.self)
        NSLog(
            "TOKEN-PROBE logits shape=%@ argmax=%d min=%.3f max=%.3f nan=%d",
            "\(last.shape)", am, minV, maxV, nanCount)
    }
}
