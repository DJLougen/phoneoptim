import XCTest

@testable import PhoneFastCore

final class PhoneFastCoreTests: XCTestCase {

    func testE2BPinsMatchManifestGeometry() {
        let pins = Gemma4TargetQuantizationPins.e2b
        XCTAssertEqual(pins.fallbackBits, 4)
        XCTAssertEqual(pins.overrideBits, 8)
        XCTAssertEqual(pins.groupSize, 64)
        XCTAssertEqual(pins.modeName, "affine")
        // 35 layers x 3 MLP families = the 105 override keys in config.json.
        XCTAssertEqual(pins.checkpointOverrideKeys().count, 105)
        XCTAssertTrue(
            pins.checkpointOverrideKeys()
                .contains("language_model.model.layers.34.mlp.up_proj"))
    }

    func testE4BPinsMatchManifestGeometry() {
        let pins = Gemma4TargetQuantizationPins.e4b
        XCTAssertEqual(pins.checkpointOverrideKeys().count, 126)
    }

    func testRevisionsAreFullSHAs() {
        for rev in [
            PhoneFastConstants.e2bRevision, PhoneFastConstants.e2bAssistantRevision,
            PhoneFastConstants.e4bRevision, PhoneFastConstants.e4bAssistantRevision,
        ] {
            XCTAssertEqual(rev.count, 40)
        }
    }
}
