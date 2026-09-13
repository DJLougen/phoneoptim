import Foundation
import PhoneFastCore
import PhoneFastHarness
import PhoneFastModel
import TrackRunnerE2B
import TrackRunnerE4B

struct BenchResult: Encodable {
    var track: String
    var weights: String
    var assistant: String?
    var image: String?
    var maxTokens: Int
    var draftDepth: Int
    var serial: Bool
    var prefillTokens: Int
    var prefillMs: Double
    var decodeTokens: Int
    var decodeMs: Double
    var tps: Double
    var proposedDraftTokens: Int?
    var acceptedDraftTokens: Int?
    var meanDraftLen: Double?
    var passthroughReason: String?
    var stopReason: String
    var output: String
}

enum BenchWorkerError: Error, LocalizedError {
    case usage(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message): return message
        }
    }
}

@main
struct BenchWorker {
    static func main() async throws {
        var weights: String?
        var assistant: String?
        var promptFile: String?
        var image: String?
        var maxTokens = 64
        var draftDepth = 0
        var serial = false
        var track = "e2b"
        var pairs = 2
        var protocolMode = false

        var args = Array(CommandLine.arguments.dropFirst())
        func takeValue(_ flag: String) throws -> String {
            guard !args.isEmpty else { throw BenchWorkerError.usage("\(flag) requires a value") }
            return args.removeFirst()
        }
        while !args.isEmpty {
            let arg = args.removeFirst()
            switch arg {
            case "--weights": weights = try takeValue(arg)
            case "--assistant": assistant = try takeValue(arg)
            case "--prompt": promptFile = try takeValue(arg)
            case "--image": image = try takeValue(arg)
            case "--max-tokens": maxTokens = Int(try takeValue(arg)) ?? maxTokens
            case "--draft-depth": draftDepth = Int(try takeValue(arg)) ?? draftDepth
            case "--track": track = try takeValue(arg)
            case "--pairs": pairs = Int(try takeValue(arg)) ?? pairs
            case "--serial": serial = true
            case "--protocol": protocolMode = true
            default: throw BenchWorkerError.usage("unknown argument '\(arg)'")
            }
        }

        guard let weights else { throw BenchWorkerError.usage("--weights is required") }
        guard let promptFile else { throw BenchWorkerError.usage("--prompt is required") }
        guard track == "e2b" || track == "e4b" else {
            throw BenchWorkerError.usage("--track must be e2b or e4b")
        }
        if !serial && !protocolMode && assistant == nil {
            throw BenchWorkerError.usage("--assistant is required unless --serial")
        }
        if serial && !protocolMode { draftDepth = 0 }
        let prompt = try String(contentsOfFile: promptFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw BenchWorkerError.usage("prompt file '\(promptFile)' is empty")
        }

        let weightsURL = URL(fileURLWithPath: weights)
        let assistantURL = serial ? nil : assistant.map { URL(fileURLWithPath: $0) }
        let imageURL = image.map { URL(fileURLWithPath: $0) }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        if protocolMode {
            // Same paired protocol the on-device harness runs: pairs of
            // [serial control, candidate at --draft-depth].
            let spec = BenchSpec(
                track: track,
                weightsDir: weights,
                assistantDir: assistant,
                promptText: prompt,
                imagePath: image,
                maxTokens: maxTokens,
                draftDepth: draftDepth,
                pairs: pairs
            )
            let result = try await BenchProtocol.run(spec: spec)
            FileHandle.standardOutput.write(try encoder.encode(result))
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)
            return
        }

        let stats: Gemma4RunStats
        if track == "e2b" {
            let runner = TrackRunnerE2B.TrackGemma4Runner()
            try await runner.load(weights: weightsURL, assistant: assistantURL)
            stats = try await runner.run(
                prompt: prompt, image: imageURL,
                maxTokens: maxTokens, draftDepth: draftDepth)
        } else {
            let runner = TrackRunnerE4B.TrackGemma4Runner()
            try await runner.load(weights: weightsURL, assistant: assistantURL)
            stats = try await runner.run(
                prompt: prompt, image: imageURL,
                maxTokens: maxTokens, draftDepth: draftDepth)
        }

        let output = BenchResult(
            track: track,
            weights: weights,
            assistant: serial ? nil : assistant,
            image: image,
            maxTokens: maxTokens,
            draftDepth: draftDepth,
            serial: serial,
            prefillTokens: stats.prefillTokens,
            prefillMs: stats.prefillMilliseconds,
            decodeTokens: stats.decodeTokens,
            decodeMs: stats.decodeMilliseconds,
            tps: stats.decodeTokensPerSecond,
            proposedDraftTokens: stats.proposedDraftTokens,
            acceptedDraftTokens: stats.acceptedDraftTokens,
            meanDraftLen: stats.meanDraftLength,
            passthroughReason: stats.passthroughReason,
            stopReason: stats.stopReason,
            output: stats.text
        )

        let data = try encoder.encode(output)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    }
}
