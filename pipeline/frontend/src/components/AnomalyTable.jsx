import React, { useState } from 'react';
import { Filter, Eye, CheckCircle2, XCircle, Info, Compass, ShieldAlert } from 'lucide-react';

export default function AnomalyTable({ detections = [], onSelectAnomaly, selectedAnomalyId }) {
  const [filterClass, setFilterClass] = useState('ALL');

  const filteredList = detections.filter((d) => {
    if (filterClass === 'ALL') return true;
    return d.classification === filterClass;
  });

  return (
    <div className="glass-panel table-panel">
      <div className="table-header-wrap">
        <div className="table-title">
          <Filter size={18} className="text-cyan" />
          <h3>Acoustic Target Inspection & Ranking Matrix</h3>
          <span className="badge-count">{filteredList.length} Targets</span>
        </div>

        {/* Filter Pills */}
        <div className="table-filter-pills">
          {['ALL', 'ghost_net', 'shipwreck', 'aircraft', 'seafloor_debris'].map((c) => (
            <button
              key={c}
              className={`pill-btn ${filterClass === c ? 'active' : ''}`}
              onClick={() => setFilterClass(c)}
            >
              {c === 'ALL' ? 'All Classes' : c.replace('_', ' ').toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Calibration Transparency Banner */}
      <div className="calibration-disclaimer-banner">
        <Info size={14} className="text-cyan" />
        <span>
          <strong>Operational Ranking Notice:</strong> Composite score (0–100) combines YOLO detector confidence and acoustic physics verification as a ranking aid. Empirical probability calibration is pending ground-truth validation.
        </span>
      </div>

      <div className="table-scroll-wrap">
        <table className="anomaly-data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>CLASSIFICATION</th>
              <th>REVIEW STATUS</th>
              <th>COMPOSITE SCORE (0–100)</th>
              <th>COMPONENT EVIDENCE</th>
              <th>DIMENSIONS (L×W)</th>
              <th>SEABED HT</th>
              <th>GEOLOCATION</th>
              <th>ACTIONABLE RECOMMENDATION</th>
            </tr>
          </thead>
          <tbody>
            {filteredList.length > 0 ? (
              filteredList.map((d) => {
                const isSelected = selectedAnomalyId === d.anomaly_id;
                const isNet = d.classification === 'ghost_net';
                const isCritical = d.priority_level === 'CRITICAL' || d.review_status === 'high_priority_review';
                const compScore = d.composite_score_pct ?? d.calibrated_confidence_pct ?? 50.0;
                const yoloScore = d.detector_score ?? d.raw_confidence ?? 0.5;
                const physicsPassed = d.physics_verified ?? d.physics?.is_verified ?? false;
                const geoQuality = d.georeference_quality || 'unnavigated_relative';

                return (
                  <tr 
                    key={d.anomaly_id} 
                    className={`anomaly-row ${isSelected ? 'selected' : ''} ${isCritical ? 'critical-row' : ''}`}
                    onClick={() => onSelectAnomaly && onSelectAnomaly(d)}
                  >
                    <td className="font-mono text-cyan">{d.anomaly_id}</td>
                    <td>
                      <span className={`class-badge ${d.classification}`}>
                        {d.classification.replace('_', ' ').toUpperCase()}
                      </span>
                    </td>
                    <td>
                      <span className={`priority-badge ${d.review_status || d.priority_level?.toLowerCase()}`}>
                        {d.review_status === 'high_priority_review'
                          ? 'HIGH PRIORITY'
                          : d.review_status === 'low_score_review'
                          ? 'LOW SCORE'
                          : 'STANDARD REVIEW'}
                      </span>
                    </td>
                    <td>
                      <div className="conf-cell">
                        <span className="conf-value font-mono">{compScore.toFixed(1)}/100</span>
                        <div className="conf-bar-bg">
                          <div 
                            className="conf-bar-fill" 
                            style={{ 
                              width: `${compScore}%`,
                              backgroundColor: compScore >= 75 ? 'var(--accent-coral)' : compScore >= 45 ? 'var(--accent-cyan)' : 'var(--text-muted)'
                            }}
                          ></div>
                        </div>
                        <span className="tier-tag">Uncalibrated Ranking</span>
                      </div>
                    </td>
                    <td>
                      <div className="evidence-cell text-xs font-mono">
                        <div>
                          <span className="text-muted">YOLO: </span>
                          <span className="text-cyan">{(yoloScore * 100).toFixed(0)}%</span>
                        </div>
                        <div>
                          <span className="text-muted">Shadow: </span>
                          {physicsPassed ? (
                            <span className="text-green font-semibold">PASS</span>
                          ) : (
                            <span className="text-coral font-semibold">FAIL</span>
                          )}
                        </div>
                        <div>
                          <span className="text-muted">Quality: </span>
                          <span className="text-amber">
                            {geoQuality === 'measured'
                              ? 'GPS Measured'
                              : geoQuality === 'estimated_dead_reckoning'
                              ? 'Dead Reckoned'
                              : 'Relative Swath'}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td className="font-mono">
                      {d.estimated_length_m}m × {d.estimated_width_m}m
                      <div className="area-sub">({d.estimated_area_m2} m²)</div>
                    </td>
                    <td className="font-mono text-amber">
                      {d.estimated_height_m}m
                      {d.towfish_roll_deg ? (
                        <div className="text-muted text-xs" title="Corrected for towfish roll">
                          Roll: {d.towfish_roll_deg > 0 ? '+' : ''}{d.towfish_roll_deg}°
                        </div>
                      ) : null}
                    </td>
                    <td className="font-mono text-xs">
                      {d.latitude != null && d.longitude != null ? (
                        <>
                          {d.latitude.toFixed(6)}° N<br />
                          {d.longitude.toFixed(6)}° E
                          <div className="text-muted" style={{ fontSize: '10px' }}>
                            ({d.gps_source})
                          </div>
                        </>
                      ) : (
                        <div className="text-muted">
                          Rel: {d.relative_east_m != null ? `${d.relative_east_m}m E` : ''}
                          <br />
                          {d.relative_north_m != null ? `${d.relative_north_m}m N` : ''}
                          <span className="text-amber" style={{ display: 'block', fontSize: '10px' }}>
                            [Unnavigated Swath]
                          </span>
                        </div>
                      )}
                    </td>
                    <td className="recommendation-cell">
                      <p className="rec-text">{d.recommended_action}</p>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={9} className="no-data-cell">
                  No anomalous targets found matching current filter criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
