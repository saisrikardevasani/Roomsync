#pragma once
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// Packet Loss Concealment stage (Stage 4).
//
// Two-tier strategy:
//   ≤40ms loss  → Opus RED / FEC (decoded from redundancy header, RFC 2198)
//   40–100ms    → LPCNet AI reconstruction
class PLCProcessor {
public:
    PLCProcessor();
    ~PLCProcessor();

    bool init(const std::string& lpcnet_onnx_path = "");

    // Submit a frame that arrived successfully.
    void push_good_frame(const float* frame, int n_samples);

    // Called when a frame was lost. Returns concealed audio.
    // loss_ms: estimated duration of loss so far (used to pick FEC vs LPCNet).
    std::vector<float> conceal(int loss_ms);

    // Attempt to decode Opus RED redundancy data for a lost frame.
    // Returns true if FEC data was available and out is filled.
    bool decode_fec(const uint8_t* red_payload, int payload_len,
                    float* out, int n_samples);

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
