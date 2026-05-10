#include "maxine_pipeline.h"
#include "../pipeline/config.h"
#include <chrono>
#include <cstring>

#ifdef ROOMSYNC_WITH_MAXINE
#include "nvAudioEffects.h"
#endif

struct MaxinePipeline::Impl {
    std::string sdk_path;
    std::chrono::steady_clock::time_point last_gpu_retry;

#ifdef ROOMSYNC_WITH_MAXINE
    NvAFX_Handle aec_h      = nullptr;
    NvAFX_Handle denoiser_h = nullptr;
    NvAFX_Handle superres_h = nullptr;
#endif

    std::vector<float> ref_buf;  // holds latest reference frame for AEC
    std::vector<float> aec_out;
    std::vector<float> denoised_out;
    std::vector<float> final_out;
};

MaxinePipeline::MaxinePipeline() : impl_(new Impl()) {
    impl_->ref_buf.assign(FRAME_SIZE, 0.0f);
    impl_->aec_out.resize(FRAME_SIZE);
    impl_->denoised_out.resize(FRAME_SIZE);
    impl_->final_out.resize(FRAME_SIZE);
    impl_->last_gpu_retry = std::chrono::steady_clock::now();
}

MaxinePipeline::~MaxinePipeline() {
#ifdef ROOMSYNC_WITH_MAXINE
    if (impl_->aec_h)      NvAFX_DestroyEffect(impl_->aec_h);
    if (impl_->denoiser_h) NvAFX_DestroyEffect(impl_->denoiser_h);
    if (impl_->superres_h) NvAFX_DestroyEffect(impl_->superres_h);
#endif
    delete impl_;
}

bool MaxinePipeline::init(const std::string& sdk_path) {
    impl_->sdk_path = sdk_path;

#ifdef ROOMSYNC_WITH_MAXINE
    NvAFX_Status status;

    // Acoustic Echo Cancellation
    status = NvAFX_CreateEffect(NVAFX_EFFECT_AEC, &impl_->aec_h);
    if (status != NVAFX_STATUS_SUCCESS) goto fallback;
    NvAFX_SetU32(impl_->aec_h, NVAFX_PARAM_AEC_ENABLE_VAD, 1);

    // Denoiser
    status = NvAFX_CreateEffect(NVAFX_EFFECT_DENOISER, &impl_->denoiser_h);
    if (status != NVAFX_STATUS_SUCCESS) goto fallback;
    NvAFX_SetF32(impl_->denoiser_h, NVAFX_PARAM_DENOISER_INTENSITY, 1.0f);

    // Super Resolution (8kHz → 16kHz bandwidth extension)
    status = NvAFX_CreateEffect(NVAFX_EFFECT_SUPER_RESOLUTION, &impl_->superres_h);
    if (status != NVAFX_STATUS_SUCCESS) goto fallback;

    // Load models into GPU
    status = NvAFX_Load(impl_->aec_h);
    if (status != NVAFX_STATUS_SUCCESS) goto fallback;
    NvAFX_Load(impl_->denoiser_h);
    NvAFX_Load(impl_->superres_h);

    gpu_active_ = true;
    return true;

fallback:
#endif
    gpu_active_ = false;
    return false;
}

void MaxinePipeline::push_reference(const float* frame, int n_samples) {
    if (n_samples > FRAME_SIZE) n_samples = FRAME_SIZE;
    std::memcpy(impl_->ref_buf.data(), frame, n_samples * sizeof(float));
}

std::vector<float> MaxinePipeline::process(const float* frame, int n_samples) {
    // Periodically retry GPU init (every GPU_RECHECK_INTERVAL_S seconds)
    if (!gpu_active_) {
        auto now     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           now - impl_->last_gpu_retry).count();
        if (elapsed >= GPU_RECHECK_INTERVAL_S) {
            impl_->last_gpu_retry = now;
            init(impl_->sdk_path);
        }
    }

#ifdef ROOMSYNC_WITH_MAXINE
    if (gpu_active_) {
        const float* mic_in = frame;
        const float* ref_in = impl_->ref_buf.data();
        float* aec_out      = impl_->aec_out.data();
        float* den_out      = impl_->denoised_out.data();
        float* fin_out      = impl_->final_out.data();

        NvAFX_Run(impl_->aec_h,      &mic_in, &ref_in, &aec_out, n_samples);
        NvAFX_Run(impl_->denoiser_h, &aec_out, nullptr, &den_out, n_samples);
        NvAFX_Run(impl_->superres_h, &den_out, nullptr, &fin_out, n_samples);

        return std::vector<float>(fin_out, fin_out + n_samples);
    }
#endif

    // CPU fallback: pass-through (CPU pipeline stages handle it)
    return std::vector<float>(frame, frame + n_samples);
}

void MaxinePipeline::try_reload_gpu() {
    if (!gpu_active_) init(impl_->sdk_path);
}
