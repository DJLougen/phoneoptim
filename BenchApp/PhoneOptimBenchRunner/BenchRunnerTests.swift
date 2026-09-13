import XCTest

import PhoneFastCore
import PhoneFastHarness
@testable import PhoneOptimBench

/// Headless driver: `xcodebuild test -destination 'id=<device>'` runs this
/// inside the host app so Documents is the app container. The result JSON is
/// written to Documents/results/ AND attached to the test.
final class BenchRunnerTests: XCTestCase {

    func testRunBenchProtocol() async throws {
        let result = try await BenchProtocol.run(spec: BenchRunner.loadSpec())

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
        XCTAssertTrue(result.outputsMatch)
    }
}
