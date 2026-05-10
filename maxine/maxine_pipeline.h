#pragma once
#include <vector>
#include <string>

// NVIDIA Maxine Audio Effects GPU pipeline.
// Wraps NvAFX_AEC + NvAFX_DENOISER + NvAFX_SUPER_RESOLUTION.
// Falls back to CPU path automatically if GPU is unavailable.
class MaxinePipeline {
public:
    MaxinePipeline();
    ~MaxinePipeline();

    // sdk_path: directory containing nvAudioEffects.h and libNvAFX.so/dll.
    // Returns true if GPU path was successfully initialized.
    bool init(const std::string& sdk_path);

    void push_reference(const float* frame, int n_samples);
    std::vector<float> process(const float* frame, int n_samples);

    bool gpu_active() const { return gpu_active_; }

private:
    struct Impl;
    Impl* impl_       = nullptr;
    bool  gpu_active_ = false;

    // Periodically re-try GPU init if it failed.
    void try_reload_gpu();
};
