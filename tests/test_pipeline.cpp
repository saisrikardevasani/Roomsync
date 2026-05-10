#include <gtest/gtest.h>
#include "../pipeline/pipeline.h"
#include "../pipeline/config.h"
#include <vector>
#include <cmath>
#include <chrono>

static std::vector<float> make_sine(float freq, int n = FRAME_SIZE) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = 0.5f * std::sin(2.0f * M_PI * freq * i / SAMPLE_RATE);
    return v;
}

// ── Pipeline init ─────────────────────────────────────────────────────────────

TEST(PipelineTest, InitSucceeds) {
    Pipeline p;
    EXPECT_TRUE(p.init(PipelineConfig{}));
}

TEST(PipelineTest, ProcessReturnsCorrectFrameSize) {
    Pipeline p;
    p.init(PipelineConfig{});
    auto mic = make_sine(1000.0f);
    p.push_reference(mic.data(), FRAME_SIZE);
    auto out = p.process(mic.data(), FRAME_SIZE);
    EXPECT_EQ((int)out.size(), FRAME_SIZE);
}

TEST(PipelineTest, PacketLossConcealment) {
    Pipeline p;
    p.init(PipelineConfig{});

    // Feed 10 good frames
    auto mic = make_sine(500.0f);
    for (int i = 0; i < 10; ++i) {
        p.push_reference(mic.data(), FRAME_SIZE);
        p.process(mic.data(), FRAME_SIZE);
    }

    // Conceal 5 frames (50ms loss)
    for (int i = 0; i < 5; ++i) {
        p.push_reference(mic.data(), FRAME_SIZE);
        auto concealed = p.process(nullptr, FRAME_SIZE, /*packet_lost=*/true);
        ASSERT_EQ((int)concealed.size(), FRAME_SIZE);
        // Concealed frame must not be all zeros after good history exists
        float energy = 0.0f;
        for (float s : concealed) energy += s * s;
        EXPECT_GT(energy, 0.0f) << "Concealed frame is silent at loss step " << i;
    }
}

TEST(PipelineTest, LatencyUnder10ms) {
    Pipeline p;
    p.init(PipelineConfig{});
    auto mic = make_sine(440.0f);

    const int N_FRAMES = 200;
    float max_ms = 0.0f;
    for (int i = 0; i < N_FRAMES; ++i) {
        p.push_reference(mic.data(), FRAME_SIZE);
        p.process(mic.data(), FRAME_SIZE);
        float ms = p.stats().pipeline_ms;
        if (ms > max_ms) max_ms = ms;
    }
    // In stub mode (no models) we expect well under 1ms; target is <10ms with models
    EXPECT_LT(max_ms, 10.0f)
        << "Pipeline latency " << max_ms << "ms exceeds 10ms target";
}

TEST(PipelineTest, ResetClearsState) {
    Pipeline p;
    p.init(PipelineConfig{});
    auto mic = make_sine(1000.0f);

    for (int i = 0; i < 20; ++i) {
        p.push_reference(mic.data(), FRAME_SIZE);
        p.process(mic.data(), FRAME_SIZE);
    }
    p.reset();

    // After reset stats should be zeroed
    auto s = p.stats();
    EXPECT_EQ(s.plc_loss_ms, 0);
}
