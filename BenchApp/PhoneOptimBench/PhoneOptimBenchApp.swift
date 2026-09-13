import PhoneFastCore
import PhoneFastHarness
import SwiftUI

@main
struct PhoneOptimBenchApp: App {
    var body: some Scene {
        WindowGroup {
            BenchView()
        }
    }
}

struct BenchView: View {
    @State private var log = "Ready. Weights must already be in Documents/."
    @State private var running = false

    var body: some View {
        VStack {
            Button(running ? "Running…" : "Run Benchmark") {
                running = true
                Task {
                    do {
                        let line = try await BenchRunner.runFromDocuments { msg in
                            Task { @MainActor in log += "\n" + msg }
                        }
                        log += "\n" + line
                    } catch {
                        log += "\nFAILED: \(error.localizedDescription)"
                    }
                    running = false
                }
            }
            .disabled(running)
            .padding()
            ScrollView {
                Text(log)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
        }
    }
}
