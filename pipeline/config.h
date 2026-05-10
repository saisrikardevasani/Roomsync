#pragma once

// Audio fundamentals
constexpr int   SAMPLE_RATE          = 16000;   // Hz
constexpr int   FRAME_SIZE_MS        = 10;       // ms per processing block
constexpr int   FRAME_SIZE           = SAMPLE_RATE * FRAME_SIZE_MS / 1000; // 160 samples
constexpr int   CHANNELS             = 1;        // mono

// AEC (Acoustic Echo Cancellation)
constexpr float AEC_ERLE_TARGET_DB   = 45.0f;
constexpr int   AEC_LATENCY_MS       = 8;
constexpr int   AEC_FILTER_LENGTH_MS = 150;

// Noise suppression
constexpr float NOISE_REDUCTION_DB   = 15.0f;
constexpr int   NOISE_LATENCY_MS     = 3;

// Source separation
constexpr int   SEP_LATENCY_MS       = 8;
constexpr int   SEP_MAX_SPEAKERS     = 4;

// Packet Loss Concealment
constexpr int   PLC_FEC_MAX_MS       = 40;    // Opus RED covers up to 40ms
constexpr int   PLC_AI_MAX_MS        = 100;   // LPCNet covers 40–100ms
constexpr int   PLC_LATENCY_MS       = 10;

// Encryption
constexpr int   FRAME_ENCRYPT_MS     = 10;
constexpr int   AES_KEY_BITS         = 256;
constexpr int   AES_KEY_BYTES        = AES_KEY_BITS / 8;
constexpr int   AES_TAG_BYTES        = 16;
constexpr int   AES_NONCE_BYTES      = 12;

// CPU degradation thresholds (fraction of one core)
constexpr float CPU_DEGRADE_THRESHOLD  = 0.80f;  // disable AI PLC above 80%
constexpr float CPU_FALLBACK_THRESHOLD = 0.95f;  // disable NRES above 95%

// Room coordinator
constexpr int   REF_BUS_SAMPLE_RATE  = 8000;   // Hz — reference bus is 8kHz
constexpr int   REF_BUS_BITRATE_KBPS = 16;
constexpr int   REF_BUS_JITTER_MS    = 20;
constexpr int   COORD_UDP_PORT       = 45320;

// GPU fallback timing
constexpr int   GPU_RECHECK_INTERVAL_S = 30;
