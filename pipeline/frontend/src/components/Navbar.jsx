import React from 'react';
import { Waves, Cpu, ShieldCheck, Activity } from 'lucide-react';

export default function Navbar({ healthData, latencyData }) {
  return (
    <header className="app-navbar">
      <div className="navbar-brand">
        <div className="brand-icon">
          <Waves size={24} className="cyan-glow" />
        </div>
        <div>
          <h1 className="brand-title">AquaScan-AI <span className="title-version">// SSS PIPELINE</span></h1>
          <p className="brand-subtitle">Automated Side-Scan Sonar Marine Debris & Ghost Net Detection Suite</p>
        </div>
      </div>

      <div className="navbar-status-group">
        <div className="status-badge live">
          <Activity size={14} className="pulse-icon" />
          <span>PIPELINE: {healthData?.pipeline ? 'ONLINE' : 'CONNECTING...'}</span>
        </div>

        <div className="status-badge gpu">
          <Cpu size={14} />
          <span>GPU: {healthData?.gpu_name || 'NVIDIA RTX 3050 (6GB)'}</span>
        </div>

        <div className="status-badge model">
          <ShieldCheck size={14} />
          <span>MODEL: YOLOv8n (99.2% mAP)</span>
        </div>
      </div>
    </header>
  );
}
