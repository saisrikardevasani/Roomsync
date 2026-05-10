//! Bridge between the Tauri UI and the native C++ audio pipeline.
//!
//! In production this calls into the roomsync_pipeline shared library via FFI.
//! The stub below provides compilable placeholder stats so the UI works without
//! the C++ pipeline built.

use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct PipelineStats {
    pub latency_ms:        f32,
    pub erle_db:           f32,
    pub noise_reduction_db: f32,
    pub packet_loss:       f32,
    pub gpu_active:        bool,
    pub peers:             Vec<String>,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            latency_ms:        0.0,
            erle_db:           0.0,
            noise_reduction_db: 0.0,
            packet_loss:       0.0,
            gpu_active:        false,
            peers:             Vec::new(),
        }
    }
}

// ── PipelineBridge ────────────────────────────────────────────────────────────

pub struct PipelineBridge {
    stats:   Arc<Mutex<PipelineStats>>,
    stop_tx: std::sync::mpsc::Sender<()>,
    _thread: JoinHandle<()>,
}

impl PipelineBridge {
    pub fn start(
        room_code: String,
        use_gpu: bool,
        _use_e2e: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stats = Arc::new(Mutex::new(PipelineStats {
            gpu_active: use_gpu,
            peers: vec![format!("laptop-{}", &room_code[..2.min(room_code.len())])],
            ..Default::default()
        }));

        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let stats_clone        = Arc::clone(&stats);
        let start              = Instant::now();

        let thread = thread::spawn(move || {
            // Simulate pipeline metrics that ramp up to steady-state
            loop {
                if stop_rx.try_recv().is_ok() { break; }

                let elapsed = start.elapsed().as_secs_f32();
                let warmup  = (elapsed / 2.0).min(1.0);

                let mut s = stats_clone.lock().unwrap();
                s.latency_ms         = 3.5 + warmup * 4.5 + (elapsed.sin() * 0.8).abs();
                s.erle_db            = warmup * 47.0;
                s.noise_reduction_db = warmup * 8.5 + (elapsed * 0.7).sin().abs() * 3.0;
                s.packet_loss        = if elapsed > 5.0 { 1.2 } else { 0.0 };
                drop(s);

                thread::sleep(Duration::from_millis(250));
            }
        });

        eprintln!("[PipelineBridge] Started for room={room_code} gpu={use_gpu}");

        Ok(Self { stats, stop_tx, _thread: thread })
    }

    pub fn stats(&self) -> PipelineStats {
        self.stats.lock().unwrap().clone()
    }

    pub fn stop(self) {
        let _ = self.stop_tx.send(());
        eprintln!("[PipelineBridge] Stopped");
    }
}
