import Foundation
import PhoneFastHarness

/// Shared driver for the UI button and the hosted XCTest. Reads
/// `Documents/bench-spec.json` when present, otherwise builds the default
/// E2B spec, runs `BenchProtocol`, and writes the result JSON into
/// `Documents/results/`.
public enum BenchRunner {

    public static func defaultSpec() -> BenchSpec {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let prompt =
            (try? String(
                contentsOf: Bundle.main.url(forResource: "short", withExtension: "txt")
                    ?? docs.appendingPathComponent("short.txt"), encoding: .utf8))
            ?? "Explain why the sky is blue in two sentences."
        return BenchSpec(
            track: "e2b",
            weightsDir: docs.appendingPathComponent("gemma-4-E2B-it-qat-4bit").path,
            assistantDir: docs.appendingPathComponent("gemma-4-E2B-it-qat-assistant-4bit").path,
            promptText: prompt.trimmingCharacters(in: .whitespacesAndNewlines),
            maxTokens: 64,
            draftDepth: 3,
            pairs: 2
        )
    }

    public static func loadSpec() throws -> BenchSpec {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let specURL = docs.appendingPathComponent("bench-spec.json")
        if let data = try? Data(contentsOf: specURL) {
            return try JSONDecoder().decode(BenchSpec.self, from: data)
        }
        return defaultSpec()
    }

    /// Runs the bench, persists JSON, returns a one-line summary.
    @discardableResult
    public static func runFromDocuments(
        progress: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> String {
        let spec = try loadSpec()
        progress(
            "spec: track=\(spec.track) depth=\(spec.draftDepth) pairs=\(spec.pairs) maxTokens=\(spec.maxTokens)"
        )
        progress("loading \(spec.weightsDir)")
        let result = try await BenchProtocol.run(spec: spec)

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let resultsDir = docs.appendingPathComponent("results", isDirectory: true)
        try FileManager.default.createDirectory(at: resultsDir, withIntermediateDirectories: true)
        let name =
            "bench-\(spec.track)-d\(spec.draftDepth)-\(Int(Date().timeIntervalSince1970)).json"
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)
        try data.write(to: resultsDir.appendingPathComponent(name))
        progress("wrote Documents/results/\(name)")

        return String(
            format:
                "composite %.3f | prefill %.2f ms/tok | decode %.1f tok/s (serial %.1f) | match=%@",
            result.composite,
            result.candidatePrefillMsPerToken,
            result.candidateDecodeMsPerToken > 0
                ? 1000.0 / result.candidateDecodeMsPerToken : 0,
            result.serialDecodeMsPerToken > 0 ? 1000.0 / result.serialDecodeMsPerToken : 0,
            result.outputsMatch ? "yes" : "NO"
        )
    }
}
