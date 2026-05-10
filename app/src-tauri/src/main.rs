// Prevents an additional console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, Mutex};
use tauri::State;

mod pipeline_bridge;

use pipeline_bridge::{PipelineBridge, PipelineStats};

// ── State ─────────────────────────────────────────────────────────────────────

type AppState = Arc<Mutex<Option<PipelineBridge>>>;

// ── Tauri commands ────────────────────────────────────────────────────────────

#[tauri::command]
fn start_pipeline(
    room_code: String,
    use_gpu: bool,
    use_e2e: bool,
    state: State<AppState>,
) -> Result<(), String> {
    let bridge = PipelineBridge::start(room_code, use_gpu, use_e2e)
        .map_err(|e| e.to_string())?;
    *state.lock().map_err(|e| e.to_string())? = Some(bridge);
    Ok(())
}

#[tauri::command]
fn stop_pipeline(state: State<AppState>) -> Result<(), String> {
    let mut guard = state.lock().map_err(|e| e.to_string())?;
    if let Some(bridge) = guard.take() {
        bridge.stop();
    }
    Ok(())
}

#[tauri::command]
fn get_pipeline_stats(state: State<AppState>) -> Result<PipelineStats, String> {
    let guard = state.lock().map_err(|e| e.to_string())?;
    guard
        .as_ref()
        .map(|b| b.stats())
        .ok_or_else(|| "Pipeline not running".to_string())
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let app_state: AppState = Arc::new(Mutex::new(None));

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            start_pipeline,
            stop_pipeline,
            get_pipeline_stats,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
