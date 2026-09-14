import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

/// Byte range and dtype of one tensor inside a safetensors file.
public struct SafetensorsEntry {
    public var offset: UInt64  // absolute file offset (header size already added)
    public var byteCount: UInt64
    public var dtype: String
    public var shape: [Int]
}

/// Parsed safetensors JSON header: tensor name -> byte range.
public struct SafetensorsIndex {
    public var entries: [String: SafetensorsEntry] = [:]

    public init(url: URL) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw LowMemoryLoadError.badSafetensors(url.lastPathComponent)
        }
        let headerLength = lengthData.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        guard headerLength > 0, headerLength <= 512 * 1024 * 1024,
            let headerData = try handle.read(upToCount: Int(headerLength)),
            let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            throw LowMemoryLoadError.badSafetensors(url.lastPathComponent)
        }
        let dataBase = 8 + headerLength
        for (name, value) in header {
            guard name != "__metadata__",
                let entry = value as? [String: Any],
                let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
                let begin = (offsets[0] as? NSNumber)?.uint64Value,
                let end = (offsets[1] as? NSNumber)?.uint64Value,
                let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int]
            else { continue }
            entries[name] = SafetensorsEntry(
                offset: dataBase + begin, byteCount: end - begin,
                dtype: dtype, shape: shape)
        }
    }
}

/// `QuantizedEmbedding` whose rows live in the checkpoint file instead of
/// memory. The ~1.35 GB `embed_tokens_per_layer` table is only ever gathered
/// by token id, so each call `pread`s just the packed rows it needs
/// (~5 KB/token) and dequantizes them in-graph. Nothing from the table is
/// ever resident. An mmap-backed `MLXArray` was tried first, but iOS Metal
/// wraps unaligned borrowed buffers incorrectly — bf16 reads came back
/// byte-swapped — so rows are copied on CPU instead.
final class DiskResidentQuantizedEmbedding: QuantizedEmbedding {
    private let handle: FileHandle
    private let weightEntry: SafetensorsEntry
    private let scalesEntry: SafetensorsEntry
    private let biasesEntry: SafetensorsEntry

    init(
        file: URL,
        weight: SafetensorsEntry, scales: SafetensorsEntry, biases: SafetensorsEntry,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) throws {
        self.handle = try FileHandle(forReadingFrom: file)
        self.weightEntry = weight
        self.scalesEntry = scales
        self.biasesEntry = biases
        // Placeholders satisfy the parameter tree; `update(parameters:)`
        // replaces them with the checkpoint's lazy arrays, which this class
        // never evaluates.
        super.init(
            weight: zeros(weight.shape, dtype: .uint32),
            scales: zeros(scales.shape, dtype: .bfloat16),
            biases: zeros(biases.shape, dtype: .bfloat16),
            groupSize: groupSize, bits: bits, mode: mode)
    }

    private func rows(_ entry: SafetensorsEntry, ids: [Int32], rowBytes: Int) -> [UInt8] {
        let rowCount = weightEntry.shape[0]
        var data = [UInt8](repeating: 0, count: ids.count * rowBytes)
        data.withUnsafeMutableBytes { dst in
            for (i, raw) in ids.enumerated() {
                let id = Int(raw)
                guard id >= 0, id < rowCount else { continue }
                _ = Darwin.pread(
                    handle.fileDescriptor, dst.baseAddress! + i * rowBytes, rowBytes,
                    off_t(entry.offset + UInt64(id * rowBytes)))
            }
        }
        return data
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let ids = x.flattened().asArray(Int32.self)

        let w = MLXArray(rows(weightEntry, ids: ids, rowBytes: weightEntry.shape[1] * 4))
            .view(dtype: .uint32).reshaped(ids.count, weightEntry.shape[1])
        let sc = MLXArray(rows(scalesEntry, ids: ids, rowBytes: scalesEntry.shape[1] * 2))
            .view(dtype: .bfloat16).reshaped(ids.count, scalesEntry.shape[1])
        let bi = MLXArray(rows(biasesEntry, ids: ids, rowBytes: biasesEntry.shape[1] * 2))
            .view(dtype: .bfloat16).reshaped(ids.count, biasesEntry.shape[1])

        return dequantized(
            w, scales: sc, biases: bi,
            groupSize: groupSize, bits: bits, mode: mode
        ).reshaped(s + [-1])
    }
}

public enum LowMemoryLoadError: Error {
    case badSafetensors(String)
    case missingTensor(String)
    case missingProcessorType
    case mmapFailed(String)
}

/// `VLMModelFactory._load` with a memory-safe weight path: safetensors arrays
/// stay lazy (a tensor's bytes are only read when the array is evaluated),
/// `embed_tokens_per_layer` tensors alias an `mmap` of the checkpoint so row
/// gathers fault in only the pages they touch, and only the text weights are
/// evaluated up front so tower weights materialize on first multimodal use
/// rather than at load.
public func loadGemma4ContextLowMemory(
    modelDirectory: URL,
    tokenizerLoader: any TokenizerLoader
) async throws -> ModelContext {
    let configurationURL = modelDirectory.appending(component: "config.json")
    let configData = try Data(contentsOf: configurationURL)
    let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)

    let model = try await VLMModelFactory.shared.typeRegistry.createModel(
        configuration: configData, modelType: baseConfig.modelType)

    var eosTokenIds = baseConfig.effectiveEOSTokenIds
    var stopStrings = Set<String>()
    if let generationData = try? Data(
        contentsOf: modelDirectory.appending(component: "generation_config.json")),
        let generationConfig = try? JSONDecoder.json5().decode(
            GenerationConfigFile.self, from: generationData)
    {
        if let ids = generationConfig.eosTokenIds?.values {
            eosTokenIds = Set(ids)
        }
        stopStrings.formUnion(generationConfig.stopStrings)
    }

    async let tokenizerTask = tokenizerLoader.load(from: modelDirectory)

    // Weight files: index weight_map when it names files that exist, else
    // conventional model*.safetensors then any safetensors present.
    let contents =
        (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
    let present = contents.filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    var weightURLs = present.filter { $0.lastPathComponent.hasPrefix("model") }
    if weightURLs.isEmpty {
        weightURLs = present.filter { $0.lastPathComponent.hasPrefix("weight") }
    }
    if weightURLs.isEmpty { weightURLs = present }
    if let indexData = try? Data(
        contentsOf: modelDirectory.appending(component: "model.safetensors.index.json")),
        let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
        let weightMap = index["weight_map"] as? [String: String]
    {
        let indexed = Set(weightMap.values).sorted()
            .map { modelDirectory.appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
        if !indexed.isEmpty { weightURLs = indexed }
    }

    // Lazy load: arrays are not materialized until evaluated.
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    var fileIndex = [String: (url: URL, index: SafetensorsIndex)]()
    for url in weightURLs {
        let (arrays, meta) = try loadArraysAndMetadata(url: url, stream: .cpu)
        let index = try SafetensorsIndex(url: url)
        for (key, array) in arrays {
            weights[key] = array
            fileIndex[key] = (url, index)
        }
        if metadata.isEmpty { metadata = meta }
    }
    weights = model.sanitize(weights: weights, metadata: metadata)

    let quantization = baseConfig.quantization
    let perLayerQuantization = baseConfig.perLayerQuantization
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            }
            return nil
        }
    }

    // Serve the ~1.35 GB `embed_tokens_per_layer` table from the checkpoint
    // file: it is only ever gathered by token id, so the module reads just
    // the rows it needs and nothing from the table is resident. The lazy
    // load arrays stay in `weights` for `update(parameters:)` verification;
    // the disk-resident module never evaluates them.
    if let pleKey = weights.keys.first(where: {
        $0.hasSuffix("embed_tokens_per_layer.weight")
    }), let source = fileIndex[pleKey] {
        let modulePath = String(pleKey.dropLast(".weight".count))
        guard let wEntry = source.index.entries["\(modulePath).weight"],
            let sEntry = source.index.entries["\(modulePath).scales"],
            let bEntry = source.index.entries["\(modulePath).biases"]
        else { throw LowMemoryLoadError.missingTensor(modulePath) }
        let q =
            perLayerQuantization?.quantization(layer: modulePath)?.asTuple
            ?? quantization?.asTuple ?? (64, 4, .affine)
        let disk = try DiskResidentQuantizedEmbedding(
            file: source.url, weight: wEntry, scales: sEntry, biases: bEntry,
            groupSize: q.0, bits: q.1, mode: q.2)
        model.update(modules: ModuleChildren.unflattened([(modulePath, disk)]))
    }

    try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    try model.prepare()

    // Realize the text tower up front so disk I/O lands in load time rather
    // than timed prefill. Towers and the PLE table stay lazy/disk-resident;
    // they only occupy memory when a multimodal prompt touches them.
    let eager = weights.filter { key, _ in
        !(key.contains("vision_tower") || key.contains("audio_tower")
            || key.contains("embed_vision") || key.contains("embed_audio")
            || key.contains("embed_tokens_per_layer"))
    }
    eval(Array(eager.values))

    let tokenizer = try await tokenizerTask

    // Processor: preprocessor_config.json or processor_config.json.
    var processorConfigURL = modelDirectory.appending(component: "preprocessor_config.json")
    if !FileManager.default.fileExists(atPath: processorConfigURL.path) {
        processorConfigURL = modelDirectory.appending(component: "processor_config.json")
    }
    let processorData = try Data(contentsOf: processorConfigURL)
    struct ProcessorClass: Decodable {
        let processorClass: String?
        enum CodingKeys: String, CodingKey { case processorClass = "processor_class" }
    }
    let processorType =
        try JSONDecoder.json5().decode(ProcessorClass.self, from: processorData).processorClass
    guard let processorType else { throw LowMemoryLoadError.missingProcessorType }
    let processor = try await VLMModelFactory.shared.processorRegistry.createModel(
        configuration: processorData, processorType: processorType, tokenizer: tokenizer)

    let modelConfig = ModelConfiguration(
        directory: modelDirectory,
        stopStrings: stopStrings, eosTokenIds: eosTokenIds)
    return ModelContext(
        configuration: modelConfig, model: model, processor: processor,
        tokenizer: tokenizer)
}
