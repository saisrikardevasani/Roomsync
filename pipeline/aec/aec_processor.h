#pragma once
#include <vector>
#include <memory>
#include <string>

// Acoustic Echo Cancellation stage.
//
// Stage 1 of the pipeline: passes mic + room-reference → returns echo-cancelled frame.
// Internally wraps WebRTC AEC3 for the linear component, then applies the
// Neural Residual Echo Suppressor (NRES, ONNX) for non-linear residual.
class AECProcessor {
public:
    AECProcessor();
    ~AECProcessor();

    // Must be called once before process(). onnx_model_path may be empty to run
    // without NRES (falls back to AEC3-only).
    bool init(const std::string& onnx_model_path = "");

    // Feed one 10ms far-end reference frame (what the speakers are playing).
    void push_reference(const float* ref_frame, int n_samples);

    // Process one 10ms near-end mic frame → returns echo-cancelled output.
    // The caller must push_reference() at least one frame before calling process().
    std::vector<float> process(const float* mic_frame, int n_samples);

    // Returns the most recently computed ERLE estimate in dB.
    float erle_db() const { return erle_db_; }

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    float erle_db_ = 0.0f;
};
