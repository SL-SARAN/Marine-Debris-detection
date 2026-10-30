import React from 'react';
import { AlertOctagon, Anchor, Gauge, Compass, Zap, HelpCircle } from 'lucide-react';

export default function KPICards({ summary, latency }) {
  const total = summary?.total_detected ?? 0;
  const criticalNets = summary?.critical_entanglement_nets ?? 0;
  const highPriority = summary?.high_priority_reviews ?? summary?.confirmed_hazards ?? 0;
  const avgComposite = (summary?.average_composite_score ?? summary?.average_calibrated_conf)
    ? `${(summary.average_composite_score ?? summary.average_calibrated_conf).toFixed(1)}/100`
    : '--';
  const fps = latency?.realtime_fps ? `${latency.realtime_fps} FPS` : '--';
  const totalMs = latency?.total_pipeline_ms ? `${latency.total_pipeline_ms.toFixed(0)} ms` : '--';

  return (
    <div className="kpi-grid">
      <div className="kpi-card card-total">
        <div className="kpi-icon-wrap cyan">
          <Anchor size={20} />
        </div>
        <div className="kpi-info">
          <span className="kpi-label">TOTAL ANOMALIES</span>
          <span className="kpi-value cyan">{total}</span>
        </div>
      </div>

      <div className="kpi-card card-critical">
        <div className="kpi-icon-wrap coral">
          <AlertOctagon size={20} />
        </div>
        <div className="kpi-info">
          <span className="kpi-label">GHOST NET HAZARDS</span>
          <span className="kpi-value coral">{criticalNets}</span>
        </div>
      </div>

      <div className="kpi-card card-confirmed">
        <div className="kpi-icon-wrap green">
          <Gauge size={20} />
        </div>
        <div className="kpi-info">
          <span className="kpi-label">HIGH PRIORITY REVIEW</span>
          <span className="kpi-value green">{highPriority}</span>
        </div>
      </div>

      <div className="kpi-card card-conf">
        <div className="kpi-icon-wrap amber">
          <Compass size={20} />
        </div>
        <div className="kpi-info">
          <span className="kpi-label" title="Score is an uncalibrated ranking aid; probability calibration is pending ground-truth validation.">
            AVG COMPOSITE SCORE <span className="kpi-sub" style={{ fontSize: '10px' }}>(UNCALIBRATED)</span>
          </span>
          <span className="kpi-value amber">{avgComposite}</span>
        </div>
      </div>

      <div className="kpi-card card-edge">
        <div className="kpi-icon-wrap cyan">
          <Zap size={20} />
        </div>
        <div className="kpi-info">
          <span className="kpi-label">EDGE LATENCY / SPEED</span>
          <span className="kpi-value cyan">{fps} <span className="kpi-sub">({totalMs})</span></span>
        </div>
      </div>
    </div>
  );
}
