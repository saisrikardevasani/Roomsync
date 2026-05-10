#include "separator.h"
#include <cstring>

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

struct SourceSeparator::Impl {
    int n_speakers = 4;
#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    Ort::Env           ort_env{ORT_LOGGING_LEVEL_WARNING, "Separator"};
    Ort::Session*      session = nullptr;
    Ort::SessionOptions opts;
    bool               model_available = false;
    ~Impl() { delete session; }
#endif
};

SourceSeparator::SourceSeparator() : impl_(std::make_unique<Impl>()) {}

SourceSeparator::~SourceSeparator() = default;

bool SourceSeparator::init([[maybe_unused]] const std::string& onnx_path, int n_speakers) {
    impl_->n_speakers = n_speakers;
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

std::vector<float> SourceSeparator::process(const float* frame, int n_samples) {
    std::vector<float> out(frame, frame + n_samples);

#ifdef ROOMSYNC_WITH_ONNXRUNTIME
    if (impl_->model_available && impl_->session) {
        // Conv-TasNet expects shape [1, 1, T] — batch=1, channel=1, time=T
        std::array<int64_t, 3> shape{1, 1, n_samples};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_t    = Ort::Value::CreateTensor<float>(
            mem, out.data(), out.size(), shape.data(), 3);

        const char* in_names[]  = {"input"};
        const char* out_names[] = {"output_0"};  // first speaker = near-end
        auto results = impl_->session->Run(Ort::RunOptions{nullptr},
                                           in_names, &in_t, 1,
                                           out_names, 1);
        float* p = results[0].GetTensorMutableData<float>();
        std::memcpy(out.data(), p, n_samples * sizeof(float));
    }
#endif

    // Fallback: pass-through (no separation without model)
    return out;
}

void SourceSeparator::reset() {}
