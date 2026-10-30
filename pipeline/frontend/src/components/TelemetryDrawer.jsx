import React from 'react';
import { Cpu, Zap, Activity, Clock, ShieldCheck } from 'lucide-react';

export default function TelemetryDrawer({ latency, speckleMetrics, totalAnomalies }) {
  if (!latency) return null;

  return (
    <div className="telemetry-bar-container">
      <div className="telemetry-inner">
        <div className="telemetry-title-block">
          <Zap size={16} className="cyan" />
          <span className="telemetry-heading">EDGE TELEMETRY BENCHMARK</span>
        </div>

        <div className="telemetry-steps-list">
          <div className="telemetry-step">
            <span className="step-label">1. Wavelet / Lee Denoise:</span>
            <span className="step-val font-mono">{latency.denoising_ms} ms</span>
          </div>

          <div className="telemetry-step">
            <span className="step-label">2. YOLOv8n Inference:</span>
            <span className="step-val font-mono cyan">{latency.inference_ms} ms</span>
          </div>

          <div className="telemetry-step">
            <span className="step-label">3. Physics Shadow Verifier:</span>
            <span className="step-val font-mono green">{latency.physics_verification_ms} ms</span>
          </div>

          <div className="telemetry-step">
            <span className="step-label">4. Calibration & Geotag:</span>
            <span className="step-val font-mono amber">{(latency.calibration_ms + latency.georeferencing_ms).toFixed(2)} ms</span>
          </div>

          <div className="telemetry-step highlight-total">
            <span className="step-label">Total Latency:</span>
            <span className="step-val font-mono text-bold">{latency.total_pipeline_ms} ms ({latency.realtime_fps} FPS)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
