import Foundation
import PhoneFastCore
import PhoneFastModel
import TrackRunnerE2B
import TrackRunnerE4B
#if os(iOS)
    import MachO
#endif

/// The benchmark spec: one prompt, one track, a draft depth for the
/// candidate legs, and a pair count. Mirrors the mlxfast paired protocol:
/// each pair is one serial-control leg and one candidate leg over the SAME
/// prompt, strictly sequential, after one shared warm-up.
public struct BenchSpec: Codable, Sendable {
    public var track: String = "e2b"
    public var weightsDir: String
    public var assistantDir: String?
    public var promptText: String
    public var imagePath: String? = nil
    public var maxTokens: Int = 64
    public var draftDepth: Int = 3
    public var pairs: Int = 2

    public init(
        track: String = "e2b",
        weightsDir: String,
        assistantDir: String? = nil,
        promptText: String,
        imagePath: String? = nil,
        maxTokens: Int = 64,
        draftDepth: Int = 3,
        pairs: Int = 2
    ) {
        self.track = track
        self.weightsDir = weightsDir
        self.assistantDir = assistantDir
        self.promptText = promptText
        self.imagePath = imagePath
        self.maxTokens = maxTokens
        self.draftDepth = draftDepth
        self.pairs = pairs
    }

    private enum CodingKeys: String, CodingKey {
        case track, weightsDir, assistantDir, promptText, imagePath
        case maxTokens, draftDepth, pairs
    }

    /// Lenient decode: any field may be omitted and falls back to the
    /// default. `weightsDir` and `promptText` remain required.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        track = try c.decodeIfPresent(String.self, forKey: .track) ?? "e2b"
        weightsDir = try c.decode(String.self, forKey: .weightsDir)
        assistantDir = try c.decodeIfPresent(String.self, forKey: .assistantDir)
        promptText = try c.decode(String.self, forKey: .promptText)
        imagePath = try c.decodeIfPresent(String.self, forKey: .imagePath)
        maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 64
        draftDepth = try c.decodeIfPresent(Int.self, forKey: .draftDepth) ?? 3
        pairs = try c.decodeIfPresent(Int.self, forKey: .pairs) ?? 2
    }
}

public struct LegRecord: Codable, Sendable {
    public var role: String  // "serial" | "candidate"
    public var pairIndex: Int
    public var prefillTokens: Int
    public var prefillMs: Double
    public var prefillMsPerToken: Double
    public var decodeTokens: Int
    public var decodeMs: Double
    public var decodeMsPerToken: Double
    public var decodeTokensPerSecond: Double
    public var meanDraftLen: Double?
    public var proposedDraftTokens: Int?
    public var acceptedDraftTokens: Int?
    public var thermalStateBefore: Int
    public var thermalStateAfter: Int
    public var physFootprintBytes: UInt64
    public var text: String
}

public struct BenchResult: Codable, Sendable {
    public var device: String
    public var osVersion: String
    public var spec: BenchSpec
    public var warmupMs: Double
    public var legs: [LegRecord]
    public var serialPrefillMsPerToken: Double
    public var candidatePrefillMsPerToken: Double
    public var serialDecodeMsPerToken: Double
    public var candidateDecodeMsPerToken: Double
    public var prefillGain: Double
    public var decodeGain: Double
    public var composite: Double
    public var outputsMatch: Bool
    public var serialText: String
    public var candidateText: String
}

public enum BenchProtocolError: Error, LocalizedError {
    case noCandidateLegs
    case noSerialLegs

    public var errorDescription: String? {
        switch self {
        case .noCandidateLegs: return "no candidate legs ran"
        case .noSerialLegs: return "no serial legs ran"
        }
    }
}

public enum BenchProtocol {

    /// Peak resident memory of this process.
    public static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return info.phys_footprint
    }

    public static func thermalState() -> Int {
        ProcessInfo.processInfo.thermalState.rawValue
    }

    /// Run the full paired protocol. Loads the model once, warms up with an
    /// 8-token generate, then runs `spec.pairs` [serial, candidate] pairs.
    public static func run(spec: BenchSpec) async throws -> BenchResult {
        let weightsURL = URL(fileURLWithPath: spec.weightsDir)
        let assistantURL = spec.assistantDir.map { URL(fileURLWithPath: $0) }
        let imageURL = spec.imagePath.map { URL(fileURLWithPath: $0) }

        // One runner instance: candidate legs use the drafter, serial legs
        // bypass it via draftDepth 0. The drafter stays resident for both.
        let runLeg: (Int) async throws -> Gemma4RunStats
        if spec.track == "e2b" {
            let runner = TrackRunnerE2B.TrackGemma4Runner()
            try await runner.load(weights: weightsURL, assistant: assistantURL)
            runLeg = { depth in
                try await runner.run(
                    prompt: spec.promptText, image: imageURL,
                    maxTokens: spec.maxTokens, draftDepth: depth)
            }
        } else {
            let runner = TrackRunnerE4B.TrackGemma4Runner()
            try await runner.load(weights: weightsURL, assistant: assistantURL)
            runLeg = { depth in
                try await runner.run(
                    prompt: spec.promptText, image: imageURL,
                    maxTokens: spec.maxTokens, draftDepth: depth)
            }
        }

        // Warm-up so Metal JIT compilation is not timed.
        let warmStart = Date()
        _ = try await runLeg(0 == spec.draftDepth ? 0 : spec.draftDepth)
        // warm-up uses the candidate arm when one exists (serial warms too —
        // Metal kernels are shared), capped short via a second serial pass is
        // unnecessary; single warm-up matches the track contract.
        let warmupMs = Date().timeIntervalSince(warmStart) * 1000

        var legs: [LegRecord] = []
        for pair in 0..<spec.pairs {
            for role in ["serial", "candidate"] {
                let depth = role == "serial" ? 0 : spec.draftDepth
                let thermalBefore = thermalState()
                let stats = try await runLeg(depth)
                let thermalAfter = thermalState()
                legs.append(
                    LegRecord(
                        role: role,
                        pairIndex: pair,
                        prefillTokens: stats.prefillTokens,
                        prefillMs: stats.prefillMilliseconds,
                        prefillMsPerToken: stats.prefillTokens > 0
                            ? stats.prefillMilliseconds / Double(stats.prefillTokens) : 0,
                        decodeTokens: stats.decodeTokens,
                        decodeMs: stats.decodeMilliseconds,
                        decodeMsPerToken: stats.decodeTokens > 0
                            ? stats.decodeMilliseconds / Double(stats.decodeTokens) : 0,
                        decodeTokensPerSecond: stats.decodeTokensPerSecond,
                        meanDraftLen: stats.meanDraftLength,
                        proposedDraftTokens: stats.proposedDraftTokens,
                        acceptedDraftTokens: stats.acceptedDraftTokens,
                        thermalStateBefore: thermalBefore,
                        thermalStateAfter: thermalAfter,
                        physFootprintBytes: physFootprint(),
                        text: stats.text
                    ))
            }
        }

        let serialLegs = legs.filter { $0.role == "serial" }
        let candLegs = legs.filter { $0.role == "candidate" }
        guard !serialLegs.isEmpty else { throw BenchProtocolError.noSerialLegs }
        guard !candLegs.isEmpty else { throw BenchProtocolError.noCandidateLegs }

        // mlxfast scoring: per role, sum per-token times over the pairs;
        // gains clamped to [0.95, 5.0]; composite = prefill^0.25 * decode^0.75.
        func msPerToken(_ legs: [LegRecord], _ ms: (LegRecord) -> Double, _ n: (LegRecord) -> Int)
            -> Double
        {
            let t = legs.reduce(0.0) { $0 + ms($1) }
            let c = legs.reduce(0) { $0 + n($1) }
            return c > 0 ? t / Double(c) : 0
        }
        let serialPrefill = msPerToken(serialLegs, { $0.prefillMs }, { $0.prefillTokens })
        let candPrefill = msPerToken(candLegs, { $0.prefillMs }, { $0.prefillTokens })
        let serialDecode = msPerToken(serialLegs, { $0.decodeMs }, { $0.decodeTokens })
        let candDecode = msPerToken(candLegs, { $0.decodeMs }, { $0.decodeTokens })

        func clampedGain(_ base: Double, _ cand: Double) -> Double {
            guard cand > 0 else { return 0.95 }
            return min(5.0, max(0.95, base / cand))
        }
        let prefillGain = clampedGain(serialPrefill, candPrefill)
        let decodeGain = clampedGain(serialDecode, candDecode)
        let composite = pow(prefillGain, 0.25) * pow(decodeGain, 0.75)

        var uts = utsname()
        uname(&uts)
        let device = withUnsafePointer(to: &uts.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }

        let serialText = serialLegs.last?.text ?? ""
        let candidateText = candLegs.last?.text ?? ""

        return BenchResult(
            device: device,
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            spec: spec,
            warmupMs: warmupMs,
            legs: legs,
            serialPrefillMsPerToken: serialPrefill,
            candidatePrefillMsPerToken: candPrefill,
            serialDecodeMsPerToken: serialDecode,
            candidateDecodeMsPerToken: candDecode,
            prefillGain: prefillGain,
            decodeGain: decodeGain,
            composite: composite,
            outputsMatch: serialText == candidateText,
            serialText: serialText,
            candidateText: candidateText
        )
    }
}
