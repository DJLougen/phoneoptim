import Foundation

/// The tensor element types the pinned target quantization uses, spelled
/// without an MLX dependency (MLX `DType` raw-value spelling).
public enum Gemma4PinnedTensorDType: String, Equatable, Sendable {
    case uint32
    case bfloat16
}

/// The Gemma 4 E-series TARGET's frozen quantization geometry, derived from
/// the pinned checkpoints' own `config.json` `quantization` blocks (NOT
/// invented): both E2B @42f62737 and E4B @0f35c6f6 declare fallback
/// `bits=4`, `group_size=64`, `mode=affine` plus a per-layer override table
/// promoting the three MLP projections (`mlp.gate_proj`, `mlp.down_proj`,
/// `mlp.up_proj`) on EVERY decoder layer to `bits=8`, `group_size=64`.
///
/// E2B: 35 layers -> 105 override keys (`language_model.model.layers.N.mlp.*`,
/// N = 0...34). The checkpoint `dtype` is `bfloat16`; MLX affine quantization
/// packs codes into `uint32` words and keeps `scales`/`biases` at the source
/// element type.
public struct Gemma4TargetQuantizationPins: Equatable, Sendable {
    /// Bit width every quantized target module carries unless its path is an
    /// override path.
    public let fallbackBits: Int

    /// Bit width the promoted MLP projection families carry, on every layer.
    public let overrideBits: Int

    /// Group size. One value: the checkpoint uses the same group size for the
    /// fallback and every override.
    public let groupSize: Int

    /// `QuantizationMode.affine`'s raw value, compared as a string so this
    /// module stays MLX-free.
    public let modeName: String

    /// Element type of the packed code tensor (`weight`).
    public let packedWeightDType: Gemma4PinnedTensorDType

    /// Element type of the `scales` and `biases` companion tensors.
    public let scaleDType: Gemma4PinnedTensorDType

    /// Bits per packed word (`UInt32` width).
    public let packedWordBits: Int

    /// Projection families promoted to `overrideBits` on every layer.
    public let overrideFamilies: [String]

    /// Decoder layer count of the checkpoint these pins describe.
    public let layerCount: Int

    public init(
        fallbackBits: Int,
        overrideBits: Int,
        groupSize: Int,
        modeName: String,
        packedWeightDType: Gemma4PinnedTensorDType,
        scaleDType: Gemma4PinnedTensorDType,
        packedWordBits: Int,
        overrideFamilies: [String],
        layerCount: Int
    ) {
        self.fallbackBits = fallbackBits
        self.overrideBits = overrideBits
        self.groupSize = groupSize
        self.modeName = modeName
        self.packedWeightDType = packedWeightDType
        self.scaleDType = scaleDType
        self.packedWordBits = packedWordBits
        self.overrideFamilies = overrideFamilies
        self.layerCount = layerCount
    }

    /// The shipped Gemma 4 E2B target's geometry (105 override keys).
    public static let e2b = Gemma4TargetQuantizationPins(
        fallbackBits: 4,
        overrideBits: 8,
        groupSize: 64,
        modeName: "affine",
        packedWeightDType: .uint32,
        scaleDType: .bfloat16,
        packedWordBits: 32,
        overrideFamilies: ["mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"],
        layerCount: 35
    )

    /// The shipped Gemma 4 E4B target's geometry (126 override keys over
    /// 42 layers).
    public static let e4b = Gemma4TargetQuantizationPins(
        fallbackBits: 4,
        overrideBits: 8,
        groupSize: 64,
        modeName: "affine",
        packedWeightDType: .uint32,
        scaleDType: .bfloat16,
        packedWordBits: 32,
        overrideFamilies: ["mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"],
        layerCount: 42
    )

    /// The checkpoint-side quantization override keys this geometry implies
    /// (`language_model.model.layers.<N>.<family>`).
    public func checkpointOverrideKeys() -> Set<String> {
        var keys: Set<String> = []
        for layer in 0..<layerCount {
            for family in overrideFamilies {
                keys.insert("language_model.model.layers.\(layer).\(family)")
            }
        }
        return keys
    }
}
