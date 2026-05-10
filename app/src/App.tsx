import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/tauri";

interface PipelineStats {
  latency_ms:         number;
  erle_db:            number;
  noise_reduction_db: number;
  packet_loss:        number;
  gpu_active:         boolean;
  peers:              string[];
}

type AppStatus = "inactive" | "active" | "error";

const DEFAULT_STATS: PipelineStats = {
  latency_ms:         0,
  erle_db:            0,
  noise_reduction_db: 0,
  packet_loss:        0,
  gpu_active:         false,
  peers:              [],
};

// Max dB shown as 100% fill on the bar; beyond this clips to full.
const NOISE_METER_MAX_DB = 20;

function NoiseMeter({ db, active }: { db: number; active: boolean }) {
  const pct = active ? Math.min(100, (db / NOISE_METER_MAX_DB) * 100) : 0;
  const level = db < 4 ? "quiet" : db < 10 ? "moderate" : "loud";

  return (
    <div className="noise-meter">
      <div className="noise-meter-header">
        <span className="noise-meter-label">Background noise</span>
        <span className={`noise-meter-value ${active ? level : ""}`}>
          {active ? `${db.toFixed(1)} dB suppressed` : "—"}
        </span>
      </div>
      <div className="noise-bar-track">
        <div
          className={`noise-bar-fill ${active ? level : ""}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [roomCode, setRoomCode]     = useState("");
  const [status, setStatus]         = useState<AppStatus>("inactive");
  const [stats, setStats]           = useState<PipelineStats>(DEFAULT_STATS);
  const [gpuEnabled, setGpuEnabled] = useState(false);
  const [e2eEnabled, setE2eEnabled] = useState(true);
  const [errorMsg, setErrorMsg]     = useState("");

  // Poll stats from Rust backend every 250ms while active
  useEffect(() => {
    if (status !== "active") return;
    const id = setInterval(async () => {
      try {
        const s = await invoke<PipelineStats>("get_pipeline_stats");
        setStats(s);
      } catch {
        // non-fatal: stats may lag on slow machines
      }
    }, 250);
    return () => clearInterval(id);
  }, [status]);

  const handleJoin = useCallback(async () => {
    if (roomCode.length < 4) return;
    try {
      await invoke("start_pipeline", {
        roomCode: roomCode.toUpperCase(),
        useGpu:   gpuEnabled,
        useE2e:   e2eEnabled,
      });
      setStatus("active");
      setErrorMsg("");
    } catch (e: any) {
      setStatus("error");
      setErrorMsg(String(e));
    }
  }, [roomCode, gpuEnabled, e2eEnabled]);

  const handleLeave = useCallback(async () => {
    try { await invoke("stop_pipeline"); } catch {}
    setStatus("inactive");
    setStats(DEFAULT_STATS);
  }, []);

  const latencyClass =
    stats.latency_ms < 5  ? "good" :
    stats.latency_ms < 10 ? "warn" : "bad";

  const erleClass =
    stats.erle_db > 45 ? "good" :
    stats.erle_db > 30 ? "warn" : "bad";

  return (
    <div className="app">
      <h1>RoomSync</h1>
      <p className="subtitle">Multi-laptop echo cancellation · E2E encrypted</p>

      <span className={`status-badge ${status}`}>
        {status === "active" ? "● Active" :
         status === "error"  ? "✕ Error"  : "○ Inactive"}
      </span>

      {/* Room code entry */}
      <div className="room-section">
        <label>Room code</label>
        <div className="room-input">
          <input
            value={roomCode}
            onChange={e => setRoomCode(e.target.value.slice(0, 8))}
            placeholder="ABC123"
            disabled={status === "active"}
            maxLength={8}
          />
          {status !== "active" ? (
            <button onClick={handleJoin} disabled={roomCode.length < 4}>
              Join
            </button>
          ) : (
            <button className="danger" onClick={handleLeave}>
              Leave
            </button>
          )}
        </div>
        {errorMsg && (
          <p style={{ color: "#d33", fontSize: "0.8rem", marginTop: "0.5rem" }}>
            {errorMsg}
          </p>
        )}
      </div>

      {/* Peer list */}
      <div className="peers-list">
        <h3>{stats.peers.length} laptop{stats.peers.length !== 1 ? "s" : ""} in room</h3>
        {stats.peers.length === 0 ? (
          <p style={{ color: "#555", fontSize: "0.8rem" }}>Waiting for others to join…</p>
        ) : (
          stats.peers.map(p => (
            <div key={p} className="peer-item">
              <div className="peer-dot" />
              {p}
            </div>
          ))
        )}
      </div>

      {/* Metrics */}
      <div className="metrics">
        <div className="metric">
          <div className="metric-label">Latency</div>
          <div className={`metric-value ${latencyClass}`}>
            {status === "active" ? `${stats.latency_ms.toFixed(1)}ms` : "—"}
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">Echo suppression</div>
          <div className={`metric-value ${erleClass}`}>
            {status === "active" ? `${stats.erle_db.toFixed(0)}dB` : "—"}
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">Packet loss</div>
          <div className={`metric-value ${stats.packet_loss > 5 ? "warn" : "good"}`}>
            {status === "active" ? `${stats.packet_loss.toFixed(1)}%` : "—"}
          </div>
        </div>
        <div className="metric">
          <div className="metric-label">Accelerator</div>
          <div className={`metric-value ${stats.gpu_active ? "good" : ""}`}>
            {status === "active" ? (stats.gpu_active ? "GPU" : "CPU") : "—"}
          </div>
        </div>
      </div>

      {/* Noise level meter */}
      <NoiseMeter db={stats.noise_reduction_db} active={status === "active"} />

      {/* Toggles */}
      <div className="toggles">
        <div className="toggle-row">
          <span>GPU acceleration (Maxine)</span>
          <input type="checkbox" checked={gpuEnabled}
                 onChange={e => setGpuEnabled(e.target.checked)}
                 disabled={status === "active"} />
        </div>
        <div className="toggle-row">
          <span>End-to-end encryption</span>
          <input type="checkbox" checked={e2eEnabled}
                 onChange={e => setE2eEnabled(e.target.checked)}
                 disabled={status === "active"} />
        </div>
      </div>
    </div>
  );
}
