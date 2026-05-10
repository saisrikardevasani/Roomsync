#pragma once
#include <vector>
#include <memory>
#include <string>

// Multi-speaker source separation stage (Stage 3).
// Detects cross-laptop voices and suppresses them, preserving the near-end speaker.
// Conv-TasNet based; causal mode required for real-time use.
class SourceSeparator {
public:
    static constexpr int MAX_SPEAKERS = 4;

    SourceSeparator();
    ~SourceSeparator();

    // onnx_path: path to separator.onnx exported from training/separation.
    // n_speakers: expected maximum number of concurrent speakers.
    bool init(const std::string& onnx_path = "", int n_speakers = MAX_SPEAKERS);

    // Returns the near-end-speaker-only frame (cross-talk suppressed).
    std::vector<float> process(const float* frame, int n_samples);

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
