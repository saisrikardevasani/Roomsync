#include "pipeline.h"
#include "config.h"
#include "aec/aec_processor.h"
#include "noise/noise_suppressor.h"
#include "separation/separator.h"
#include "plc/plc_processor.h"
#include <chrono>
#include <mutex>
#include <cassert>

#ifdef ROOMSYNC_WITH_MAXINE
#include "../maxine/maxine_pipeline.h"
#endif

struct Pipeline::Impl {
    AECProcessor    aec;
    NoiseSuppressor noise;
    SourceSeparator sep;
    PLCProcessor    plc;

    FrameStats      last_stats;
    std::mutex      stats_mu;

    int  plc_loss_ms     = 0;
    bool gpu_active      = false;

#ifdef ROOMSYNC_WITH_MAXINE
    MaxinePipeline maxine;
#endif
};

Pipeline::Pipeline()  : impl_(std::make_unique<Impl>()) {}
Pipeline::~Pipeline() = default;

bool Pipeline::init(const PipelineConfig& cfg) {
#ifdef ROOMSYNC_WITH_MAXINE
    if (cfg.use_maxine) {
        impl_->gpu_active = impl_->maxine.init("");
    }
#endif

    if (!impl_->gpu_active) {
        if (!impl_->aec.init(cfg.aec_nres_onnx))   return false;
        if (!impl_->noise.init(cfg.noise_onnx))     return false;
        if (!impl_->sep.init(cfg.separator_onnx))   return false;
        if (!impl_->plc.init(cfg.lpcnet_onnx))      return false;
    }
    return true;
}

void Pipeline::push_reference(const float* frame, int n_samples) {
    assert(n_samples == FRAME_SIZE);
#ifdef ROOMSYNC_WITH_MAXINE
    if (impl_->gpu_active) {
        impl_->maxine.push_reference(frame, n_samples);
        return;
    }
#endif
    impl_->aec.push_reference(frame, n_samples);
}

std::vector<float> Pipeline::process(const float* frame, int n_samples, bool packet_lost) {
    assert(n_samples == FRAME_SIZE);
    auto t0 = std::chrono::steady_clock::now();

    std::vector<float> out;

    if (packet_lost) {
        impl_->plc_loss_ms += FRAME_SIZE_MS;
        out = impl_->plc.conceal(impl_->plc_loss_ms);
    } else {
        impl_->plc_loss_ms = 0;

#ifdef ROOMSYNC_WITH_MAXINE
        if (impl_->gpu_active) {
            out = impl_->maxine.process(frame, n_samples);
        } else {
#endif
            // ── Stage 1: AEC ────────────────────────────────────────────────
            out = impl_->aec.process(frame, n_samples);

            // ── Stage 2: Noise suppression ───────────────────────────────────
            out = impl_->noise.process(out.data(), (int)out.size());

            // ── Stage 3: Source separation ───────────────────────────────────
            out = impl_->sep.process(out.data(), (int)out.size());

#ifdef ROOMSYNC_WITH_MAXINE
        }
#endif
        // ── Stage 4: Feed good frame to PLC history ──────────────────────────
        impl_->plc.push_good_frame(out.data(), (int)out.size());
    }

    auto t1 = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    {
        std::lock_guard<std::mutex> lock(impl_->stats_mu);
        impl_->last_stats.erle_db         = impl_->aec.erle_db();
        impl_->last_stats.noise_reduction  = impl_->noise.reduction_db();
        impl_->last_stats.pipeline_ms      = ms;
        impl_->last_stats.plc_loss_ms      = impl_->plc_loss_ms;
    }

    return out;
}

FrameStats Pipeline::stats() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mu);
    return impl_->last_stats;
}

void Pipeline::reset() {
    impl_->aec.reset();
    impl_->noise.reset();
    impl_->sep.reset();
    impl_->plc.reset();
    impl_->plc_loss_ms = 0;
    std::lock_guard<std::mutex> lock(impl_->stats_mu);
    impl_->last_stats = FrameStats{};
}

bool Pipeline::is_gpu_active() const { return impl_->gpu_active; }
