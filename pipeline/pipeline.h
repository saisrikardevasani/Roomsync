#pragma once
#include <vector>
#include <memory>
#include <string>

struct PipelineConfig {
    std::string aec_nres_onnx;       // path to NRES onnx (empty = AEC3-only)
    std::string noise_onnx;          // path to RNNoise-GRU onnx (empty = fallback)
    std::string separator_onnx;      // path to separator.onnx (empty = passthrough)
    std::string lpcnet_onnx;         // path to lpcnet.onnx (empty = pitch-repeat PLC)
    bool        use_maxine = false;  // attempt GPU acceleration via Maxine
    int         sample_rate = 16000;
};

struct FrameStats {
    float erle_db         = 0.0f;
    float noise_reduction = 0.0f;
    float pipeline_ms     = 0.0f;  // wall-clock latency of last frame
    int   plc_loss_ms     = 0;
};

// Thread-safe, real-time audio pipeline.
// All process() calls must arrive at the real-time audio callback rate (every 10ms).
class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineConfig& cfg);

    // Feed far-end reference (what the room's speakers are playing).
    void push_reference(const float* frame, int n_samples);

    // Process one 10ms near-end mic frame.
    // If frame is nullptr the pipeline synthesises a concealed frame (packet loss).
    std::vector<float> process(const float* frame, int n_samples, bool packet_lost = false);

    FrameStats stats() const;
    void reset();
    bool is_gpu_active() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
