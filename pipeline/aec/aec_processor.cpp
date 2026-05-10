#include "aec_processor.h"
#include "../config.h"
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// NLMS adaptive filter — 32ms tail at 16 kHz is sufficient for most rooms.
static constexpr int   NLMS_LEN   = 512;
static constexpr float NLMS_MU    = 0.04f;   // step size
static constexpr float NLMS_DELTA = 1e-5f;   // regularisation (avoids div-by-0)

struct AECProcessor::Impl {
    // Far-end ring buffer: AEC_FILTER_LENGTH_MS worth of reference samples.
    // Must be at least NLMS_LEN + FRAME_SIZE to cover the full filter tap span.
    std::vector<float> ref_buf;
    // NLMS filter weights.
    std::vector<float> w;

    float echo_power  = 1e-10f;
    float error_power = 1e-10f;

    bool nres_available = false;
#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    Ort::Env            ort_env{ORT_LOGGING_LEVEL_WARNING, "NRES"};
    Ort::Session*       nres_session = nullptr;
    Ort::SessionOptions session_opts;
    ~Impl() { delete nres_session; }
#endif
};

AECProcessor::AECProcessor() : impl_(std::make_unique<Impl>()) {}
AECProcessor::~AECProcessor() = default;

bool AECProcessor::init([[maybe_unused]] const std::string& onnx_model_path) {
    // Buffer large enough so NLMS can look back NLMS_LEN taps from the oldest
    // sample of the current frame.
    const int buf_size = std::max(AEC_FILTER_LENGTH_MS * SAMPLE_RATE / 1000,
                                  NLMS_LEN + FRAME_SIZE);
    impl_->ref_buf.assign(buf_size, 0.0f);
    impl_->w.assign(NLMS_LEN, 0.0f);

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    if (!onnx_model_path.empty()) {
        impl_->session_opts.SetIntraOpNumThreads(1);
        impl_->session_opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        impl_->nres_session = new Ort::Session(
            impl_->ort_env, onnx_model_path.c_str(), impl_->session_opts);
        impl_->nres_available = true;
    }
#endif
    return true;
}

void AECProcessor::push_reference(const float* ref_frame, int n_samples) {
    assert(n_samples == FRAME_SIZE);
    auto& buf = impl_->ref_buf;
    // Slide buffer left and append new frame at the end.
    std::memmove(buf.data(), buf.data() + n_samples,
                 (buf.size() - n_samples) * sizeof(float));
    std::memcpy(buf.data() + buf.size() - n_samples,
                ref_frame, n_samples * sizeof(float));

    float pwr = 0.0f;
    for (int i = 0; i < n_samples; ++i) pwr += ref_frame[i] * ref_frame[i];
    impl_->echo_power = 0.95f * impl_->echo_power + 0.05f * (pwr / n_samples);
}

std::vector<float> AECProcessor::process(const float* mic_frame, int n_samples) {
    assert(n_samples == FRAME_SIZE);
    const auto& x = impl_->ref_buf;
    if (static_cast<int>(x.size()) < n_samples)
        return std::vector<float>(mic_frame, mic_frame + n_samples);

    std::vector<float> out(n_samples);
    auto& w       = impl_->w;
    const int N   = static_cast<int>(x.size());

    // Estimate tap-vector energy from the current reference frame.
    // Using history taps fails on frame-0 (all zeros → step size → ∞ → diverge).
    // Assuming stationarity: E[||x||²] ≈ (E_frame / n_samples) * NLMS_LEN.
    float x_energy = 0.0f;
    for (int n = 0; n < n_samples; ++n) {
        const float xi = x[N - n_samples + n];
        x_energy += xi * xi;
    }
    x_energy = (x_energy / n_samples) * NLMS_LEN;
    const float mu_n = NLMS_MU / (x_energy + NLMS_DELTA);

    float error_pwr_sum = 0.0f;
    for (int n = 0; n < n_samples; ++n) {
        // Echo estimate: inner product of weights and reference tap vector.
        // x[N - n_samples + n - k] is reference sample delayed by k taps.
        float echo_est = 0.0f;
        for (int k = 0; k < NLMS_LEN; ++k)
            echo_est += w[k] * x[N - n_samples + n - k];

        const float e = mic_frame[n] - echo_est;
        out[n] = e;
        error_pwr_sum += e * e;

        // NLMS weight update.
        for (int k = 0; k < NLMS_LEN; ++k)
            w[k] += mu_n * e * x[N - n_samples + n - k];
    }

    impl_->error_power =
        0.95f * impl_->error_power + 0.05f * (error_pwr_sum / n_samples);

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    if (impl_->nres_available && impl_->nres_session) {
        std::array<int64_t, 2> shape{1, FRAME_SIZE};
        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_t = Ort::Value::CreateTensor<float>(
            mem, out.data(), out.size(), shape.data(), 2);
        const char* in_names[]  = {"input"};
        const char* out_names[] = {"output"};
        auto res = impl_->nres_session->Run(Ort::RunOptions{nullptr},
                                            in_names, &in_t, 1, out_names, 1);
        std::memcpy(out.data(), res[0].GetTensorMutableData<float>(),
                    n_samples * sizeof(float));
    }
#endif

    if (impl_->echo_power > 1e-10f)
        erle_db_ = 10.0f * std::log10(
            impl_->echo_power / (impl_->error_power + 1e-10f));

    return out;
}

void AECProcessor::reset() {
    std::fill(impl_->ref_buf.begin(), impl_->ref_buf.end(), 0.0f);
    std::fill(impl_->w.begin(), impl_->w.end(), 0.0f);
    impl_->echo_power  = 1e-10f;
    impl_->error_power = 1e-10f;
    erle_db_           = 0.0f;
}
