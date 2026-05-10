#include "plc_processor.h"
#include "../config.h"
#include <deque>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

static constexpr int HISTORY_FRAMES = 200 / FRAME_SIZE_MS;

// LPC parameters for Tier-3 synthesis concealment.
static constexpr int   LPC_ORDER   = 12;   // standard for narrowband speech
static constexpr int   LPC_HISTORY = 320;  // 20ms context (2 frames)

// ── Levinson-Durbin for LPC coefficient estimation ────────────────────────────
// R : biased autocorrelation [0..LPC_ORDER], size LPC_ORDER+1
// a : output LPC coefficients [0..LPC_ORDER-1] (0-indexed)
// Returns prediction error variance.
static float levinson_durbin(const std::vector<float>& R, std::vector<float>& a) {
    const int p = static_cast<int>(a.size());
    std::fill(a.begin(), a.end(), 0.0f);

    float err = R[0];
    if (err < 1e-12f) return err;

    std::vector<float> prev(p, 0.0f);
    for (int i = 0; i < p; ++i) {
        float ki = R[i + 1];
        for (int j = 0; j < i; ++j)
            ki += a[j] * R[i - j];
        ki = -ki / err;
        // Clamp reflection coefficient for a stable all-pole filter.
        ki = std::max(-0.999f, std::min(0.999f, ki));

        prev = a;
        a[i] = ki;
        for (int j = 0; j < i; ++j)
            a[j] = prev[j] + ki * prev[i - 1 - j];

        err *= (1.0f - ki * ki);
        if (err <= 1e-12f) break;
    }
    return err;
}

// Synthesise one frame via LPC all-pole filter with zero excitation.
// hist : last LPC_ORDER samples from the history (oldest first).
// a    : LPC coefficients (0=lag-1, 1=lag-2, ...).
static std::vector<float> lpc_synthesise(
        const std::vector<float>& hist,
        const std::vector<float>& a,
        int n_samples) {
    std::vector<float> out(n_samples);
    // Working buffer: history + synthesised samples.
    std::vector<float> buf(hist.end() - LPC_ORDER, hist.end());
    buf.resize(LPC_ORDER + n_samples, 0.0f);

    for (int n = 0; n < n_samples; ++n) {
        float s = 0.0f;
        for (int k = 0; k < LPC_ORDER; ++k)
            s -= a[k] * buf[LPC_ORDER + n - 1 - k];
        buf[LPC_ORDER + n] = s;
        out[n] = s;
    }
    return out;
}

// ── PLCProcessor implementation ───────────────────────────────────────────────

struct PLCProcessor::Impl {
    std::deque<std::vector<float>> history;

    bool lpcnet_available = false;
#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    Ort::Env            ort_env{ORT_LOGGING_LEVEL_WARNING, "LPCNet"};
    Ort::Session*       session = nullptr;
    Ort::SessionOptions opts;
    ~Impl() { delete session; }
#endif
};

PLCProcessor::PLCProcessor() : impl_(std::make_unique<Impl>()) {}
PLCProcessor::~PLCProcessor() = default;

bool PLCProcessor::init([[maybe_unused]] const std::string& lpcnet_onnx_path) {
#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    if (!lpcnet_onnx_path.empty()) {
        impl_->opts.SetIntraOpNumThreads(1);
        impl_->opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        impl_->session = new Ort::Session(
            impl_->ort_env, lpcnet_onnx_path.c_str(), impl_->opts);
        impl_->lpcnet_available = true;
    }
#endif
    return true;
}

void PLCProcessor::push_good_frame(const float* frame, int n_samples) {
    impl_->history.emplace_back(frame, frame + n_samples);
    if (static_cast<int>(impl_->history.size()) > HISTORY_FRAMES)
        impl_->history.pop_front();
}

std::vector<float> PLCProcessor::conceal(int loss_ms) {
    const int n_samples = FRAME_SIZE;

    // ── Tier 2: LPCNet AI reconstruction (requires ONNX model) ──────────────
#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    if (impl_->lpcnet_available && impl_->session &&
        loss_ms >= PLC_FEC_MAX_MS && !impl_->history.empty()) {
        std::vector<float> ctx(HISTORY_FRAMES * FRAME_SIZE, 0.0f);
        int idx = 0;
        for (const auto& f : impl_->history) {
            int offset = idx++ * FRAME_SIZE;
            std::memcpy(ctx.data() + offset, f.data(),
                        std::min((int)f.size(), FRAME_SIZE) * sizeof(float));
        }
        std::array<int64_t, 3> ctx_shape{1, HISTORY_FRAMES, FRAME_SIZE};
        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value ctx_t = Ort::Value::CreateTensor<float>(
            mem, ctx.data(), ctx.size(), ctx_shape.data(), 3);
        const char* in_names[]  = {"context"};
        const char* out_names[] = {"concealed"};
        auto results = impl_->session->Run(Ort::RunOptions{nullptr},
                                           in_names, &ctx_t, 1, out_names, 1);
        float* p = results[0].GetTensorMutableData<float>();
        return std::vector<float>(p, p + n_samples);
    }
#endif

    // ── Tier 3: LPC synthesis (works without any external model) ─────────────
    if (!impl_->history.empty()) {
        // Flatten the last LPC_HISTORY samples from the ring buffer.
        std::vector<float> flat;
        flat.reserve(LPC_HISTORY);
        for (auto it = impl_->history.rbegin();
             it != impl_->history.rend() && (int)flat.size() < LPC_HISTORY; ++it) {
            for (auto rit = it->rbegin();
                 rit != it->rend() && (int)flat.size() < LPC_HISTORY; ++rit)
                flat.push_back(*rit);
        }
        std::reverse(flat.begin(), flat.end());  // chronological order

        // Biased autocorrelation R[0..LPC_ORDER].
        const int M = static_cast<int>(flat.size());
        std::vector<float> R(LPC_ORDER + 1, 0.0f);
        for (int lag = 0; lag <= LPC_ORDER; ++lag)
            for (int n = 0; n < M - lag; ++n)
                R[lag] += flat[n] * flat[n + lag];
        if (M > 0) for (auto& r : R) r /= M;

        // Solve LPC via Levinson-Durbin.
        std::vector<float> a(LPC_ORDER, 0.0f);
        levinson_durbin(R, a);

        // Synthesise the missing frame.
        auto out = lpc_synthesise(flat, a, n_samples);

        // Fade out as concealment duration grows to avoid a hard transition.
        const float fade = std::max(0.0f,
                           1.0f - static_cast<float>(loss_ms) / PLC_AI_MAX_MS);
        for (auto& s : out) s *= fade;

        return out;
    }

    return std::vector<float>(n_samples, 0.0f);
}

bool PLCProcessor::decode_fec(const uint8_t* /*red_payload*/, int /*payload_len*/,
                               float* out, int n_samples) {
    (void)out;
    (void)n_samples;
    return false;
}

void PLCProcessor::reset() {
    impl_->history.clear();
}
