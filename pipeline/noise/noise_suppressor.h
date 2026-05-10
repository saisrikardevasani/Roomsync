#pragma once
#include <vector>
#include <memory>
#include <string>

// Noise suppression stage (Stage 2).
// Wraps the RNNoise-derived GRU model. Falls back to a simple spectral
// subtraction when the ONNX model is not available.
class NoiseSuppressor {
public:
    NoiseSuppressor();
    ~NoiseSuppressor();

    // weights_path: path to rnnoise_weights.c compiled into a shared lib, OR
    // onnx_path: path to an ONNX export of the GRU model.
    // Either may be empty — suppressor will run in fallback mode.
    bool init(const std::string& onnx_path = "");

    // Returns a noise-suppressed frame (in-place safe).
    std::vector<float> process(const float* frame, int n_samples);

    // Estimated noise reduction applied on the last frame, in dB.
    float reduction_db() const { return reduction_db_; }

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    float reduction_db_ = 0.0f;
};
