import Foundation

/// Pinned checkpoint coordinates. The manifest files under `fixtures/`
/// carry the per-file sha256/size records for the same revisions.
public enum PhoneFastConstants {
    public static let e2bRepository = "mlx-community/gemma-4-E2B-it-qat-4bit"
    public static let e2bRevision = "42f62737af7a9fd8c1d55d79666c1a217be4e2e2"

    public static let e2bAssistantRepository = "mlx-community/gemma-4-E2B-it-qat-assistant-4bit"
    public static let e2bAssistantRevision = "7acb8d02364777be103dbf8c8dcb007f55f717e6"

    public static let e4bRepository = "mlx-community/gemma-4-E4B-it-qat-4bit"
    public static let e4bRevision = "0f35c6f6d386f7f74e628bd7c6526ce531212300"

    public static let e4bAssistantRepository = "mlx-community/gemma-4-E4B-it-qat-assistant-4bit"
    public static let e4bAssistantRevision = "dac8da1a066d5e00bc085e281fa73f2d5ec2fbe0"

    /// Directory setup.sh downloads into, relative to the repo root.
    public static let referenceWeightsDirectory = "reference_weights"
}
