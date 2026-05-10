#include "noise_suppressor.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <complex>
#include <vector>

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
#include <numeric>
#include <onnxruntime_cxx_api.h>
#endif

// ── Inline radix-2 Cooley-Tukey FFT ──────────────────────────────────────────

static void radix2_fft(std::vector<std::complex<float>>& x) {
    const int n = static_cast<int>(x.size());
    // Bit-reversal permutation.
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    // Butterfly stages.
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * static_cast<float>(M_PI) / len;
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                const auto u = x[i + j];
                const auto v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

static void radix2_ifft(std::vector<std::complex<float>>& x) {
    for (auto& c : x) c = std::conj(c);
    radix2_fft(x);
    const float inv = 1.0f / static_cast<float>(x.size());
    for (auto& c : x) c = std::conj(c) * inv;
}

// ── Spectral subtraction constants ───────────────────────────────────────────

// FFT size: next power-of-2 ≥ 2×FRAME_SIZE prevents time-domain aliasing.
static constexpr int   FFT_SIZE       = 512;
static constexpr int   HALF_FFT       = FFT_SIZE / 2 + 1;  // 257 real bins
static constexpr float OVER_SUBTRACT  = 1.5f;   // β: Boll 1979 over-subtraction
static constexpr float SPEC_FLOOR     = 0.05f;  // gain floor (prevents musical noise)
static constexpr float NOISE_ALPHA    = 0.95f;  // noise PSD EMA weight
static constexpr float VAD_SNR_THRESH = 3.0f;   // dB: below this → noise-only frame

// ── NoiseSuppressor implementation ───────────────────────────────────────────

static constexpr int INIT_FRAMES = 5;  // unconditional noise bootstrap frames

struct NoiseSuppressor::Impl {
    std::vector<float> noise_psd;  // per-bin noise PSD estimate [HALF_FFT]
    std::vector<float> hann_win;   // analysis window [FRAME_SIZE]
    int init_count = 0;            // counts frames during cold-start bootstrap

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    Ort::Env            ort_env{ORT_LOGGING_LEVEL_WARNING, "NoiseSup"};
    Ort::Session*       session         = nullptr;
    Ort::SessionOptions opts;
    bool                model_available = false;
    ~Impl() { delete session; }
#endif
};

NoiseSuppressor::NoiseSuppressor() : impl_(std::make_unique<Impl>()) {
    impl_->noise_psd.assign(HALF_FFT, 1e-6f);
    impl_->hann_win.resize(160);  // FRAME_SIZE at 16kHz / 10ms
    const int W = static_cast<int>(impl_->hann_win.size());
    for (int i = 0; i < W; ++i)
        impl_->hann_win[i] =
            0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (W - 1)));
}

NoiseSuppressor::~NoiseSuppressor() = default;

bool NoiseSuppressor::init([[maybe_unused]] const std::string& onnx_path) {
#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    if (!onnx_path.empty()) {
        impl_->opts.SetIntraOpNumThreads(1);
        impl_->opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        impl_->session = new Ort::Session(
            impl_->ort_env, onnx_path.c_str(), impl_->opts);
        impl_->model_available = true;
    }
#endif
    return true;
}

std::vector<float> NoiseSuppressor::process(const float* frame, int n_samples) {
    std::vector<float> out(frame, frame + n_samples);

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    if (impl_->model_available && impl_->session) {
        std::array<int64_t, 2> shape{1, n_samples};
        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_t = Ort::Value::CreateTensor<float>(
            mem, out.data(), out.size(), shape.data(), 2);
        const char* in_names[]  = {"input"};
        const char* out_names[] = {"output"};
        auto results = impl_->session->Run(Ort::RunOptions{nullptr},
                                           in_names, &in_t, 1, out_names, 1);
        float* p = results[0].GetTensorMutableData<float>();
        std::memcpy(out.data(), p, n_samples * sizeof(float));

        float in_pwr  = std::inner_product(frame, frame + n_samples, frame, 0.0f);
        float out_pwr = std::inner_product(out.begin(), out.end(), out.begin(), 0.0f);
        if (out_pwr > 1e-12f && in_pwr > 1e-12f)
            reduction_db_ = 10.0f * std::log10(in_pwr / out_pwr);
        return out;
    }
#endif

    // ── Spectral subtraction (Boll 1979 with over-subtraction + spectral floor) ─

    // 1. Window the input and zero-pad to FFT_SIZE.
    std::vector<std::complex<float>> X(FFT_SIZE, {0.0f, 0.0f});
    const int win_len = std::min(n_samples, static_cast<int>(impl_->hann_win.size()));
    for (int i = 0; i < win_len; ++i)
        X[i] = {frame[i] * impl_->hann_win[i], 0.0f};

    // 2. Forward FFT.
    radix2_fft(X);

    // 3. Power spectrum and total frame energy.
    float frame_energy = 0.0f;
    std::vector<float> power(HALF_FFT);
    for (int k = 0; k < HALF_FFT; ++k) {
        power[k]     = std::norm(X[k]);  // |X[k]|^2
        frame_energy += power[k];
    }

    // 4. Update noise PSD.
    // During the first INIT_FRAMES frames the estimate is uninitialized
    // (noise_psd ≈ 1e-6), so any real signal produces enormous apparent SNR
    // and the VAD never fires.  Bootstrap by updating unconditionally for the
    // first few frames regardless of VAD decision.
    bool is_noise;
    if (impl_->init_count < INIT_FRAMES) {
        is_noise = true;
        ++impl_->init_count;
    } else {
        float noise_energy = 0.0f;
        for (int k = 0; k < HALF_FFT; ++k) noise_energy += impl_->noise_psd[k];
        const float snr_db = 10.0f * std::log10((frame_energy + 1e-12f) /
                                                 (noise_energy + 1e-12f));
        is_noise = (snr_db < VAD_SNR_THRESH);
    }

    if (is_noise) {
        for (int k = 0; k < HALF_FFT; ++k)
            impl_->noise_psd[k] = NOISE_ALPHA * impl_->noise_psd[k]
                                 + (1.0f - NOISE_ALPHA) * power[k];
    }

    // 5. Wiener-like gain per bin.
    float out_pwr_total = 0.0f;
    for (int k = 0; k < HALF_FFT; ++k) {
        const float pwr    = power[k] + 1e-12f;
        const float noise  = impl_->noise_psd[k];
        const float gain   = std::max(SPEC_FLOOR,
                             std::sqrt(std::max(0.0f,
                                       1.0f - OVER_SUBTRACT * noise / pwr)));
        X[k] *= gain;
        out_pwr_total += gain * gain * power[k];

        // Restore conjugate symmetry for the IFFT.
        if (k > 0 && k < FFT_SIZE / 2)
            X[FFT_SIZE - k] = std::conj(X[k]);
    }
    // Nyquist bin must be real.
    X[FFT_SIZE / 2] = {X[FFT_SIZE / 2].real(), 0.0f};

    // 6. Inverse FFT → real output.
    radix2_ifft(X);

    for (int i = 0; i < n_samples; ++i)
        out[i] = X[i].real();

    if (frame_energy > 1e-12f)
        reduction_db_ = 10.0f * std::log10(
            (frame_energy + 1e-12f) / (out_pwr_total + 1e-12f));

    return out;
}

void NoiseSuppressor::reset() {
    std::fill(impl_->noise_psd.begin(), impl_->noise_psd.end(), 1e-6f);
    impl_->init_count = 0;
    reduction_db_ = 0.0f;
}
