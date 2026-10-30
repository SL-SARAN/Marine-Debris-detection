import React, { useState } from 'react';
import { Eye, Layers, Sparkles, Crosshair, CheckCircle2, AlertTriangle } from 'lucide-react';

export default function PipelineViewer({ stages, speckleMetrics, imageShape, detections, isLoading }) {
  const [activeTab, setActiveTab] = useState('overlay'); // 'raw', 'denoised', 'overlay'

  const rawB64 = stages?.raw_sonar_b64;
  const denoisedB64 = stages?.denoised_sonar_b64;
  const overlayB64 = stages?.detections_overlay_b64;

  const currentImage = 
    activeTab === 'raw' ? rawB64 :
    activeTab === 'denoised' ? denoisedB64 :
    overlayB64 || rawB64;

  return (
    <div className="pipeline-viewer-card">
      <div className="viewer-header">
        <div className="viewer-title-group">
          <Layers size={18} className="cyan" />
          <h2 className="viewer-title">4-STAGE ACOUSTIC PIPELINE INSPECTOR</h2>
        </div>

        {/* Tab Controls */}
        <div className="viewer-tabs">
          <button 
            className={`tab-btn ${activeTab === 'raw' ? 'active' : ''}`}
            onClick={() => setActiveTab('raw')}
          >
            <Eye size={14} />
            <span>1. RAW SWATH</span>
          </button>
          
          <button 
            className={`tab-btn ${activeTab === 'denoised' ? 'active' : ''}`}
            onClick={() => setActiveTab('denoised')}
          >
            <Sparkles size={14} />
            <span>2. WAVELET / LEE FILTER</span>
          </button>

          <button 
            className={`tab-btn ${activeTab === 'overlay' ? 'active' : ''}`}
            onClick={() => setActiveTab('overlay')}
          >
            <Crosshair size={14} />
            <span>3. AI DETECTIONS & PHYSICS</span>
          </button>
        </div>
      </div>

      {/* Metrics Strip */}
      <div className="viewer-metrics-strip">
        <div className="metric-tag">
          <span className="metric-tag-label">SWATH DIMENSIONS:</span>
          <span className="metric-tag-val">{imageShape ? `${imageShape[0]} × ${imageShape[1]} px` : '--'}</span>
        </div>

        {speckleMetrics && (
          <>
            <div className="metric-tag">
              <span className="metric-tag-label">SPECKLE ENL:</span>
              <span className="metric-tag-val cyan">{speckleMetrics.enl?.toFixed(2)}</span>
            </div>
            <div className="metric-tag">
              <span className="metric-tag-label">FILTER PSNR:</span>
              <span className="metric-tag-val green">{speckleMetrics.psnr_db?.toFixed(2)} dB</span>
            </div>
          </>
        )}

        <div className="metric-tag stage-desc">
          {activeTab === 'raw' && <span className="amber">Stage 1: Direct Port & Starboard Transducer Backscatter Swath</span>}
          {activeTab === 'denoised' && <span className="green">Stage 2: 2D Wavelet Decomposition (DWT) + Enhanced Lee Adaptive Filter</span>}
          {activeTab === 'overlay' && <span className="cyan">Stage 3: YOLOv8n Bounding Boxes + Highlight-Shadow Physics Coupling</span>}
        </div>
      </div>

      {/* Main Image Display */}
      <div className="viewer-image-container">
        {isLoading ? (
          <div className="viewer-loading-overlay">
            <div className="sonar-radar-pulse"></div>
            <p className="loading-text">INGESTING SWATH & EXECUTING 4-MODULE CV PIPELINE...</p>
          </div>
        ) : currentImage ? (
          <div className="image-wrapper">
            <img 
              src={currentImage} 
              alt="Sonar Acoustic Swath" 
              className="sonar-display-img" 
            />
            {activeTab === 'overlay' && detections && detections.length > 0 && (
              <div className="detection-overlay-legend">
                <div className="legend-item"><span className="dot net"></span> Ghost Net ({detections.filter(d=>d.classification==='ghost_net').length})</div>
                <div className="legend-item"><span className="dot wreck"></span> Shipwreck ({detections.filter(d=>d.classification==='shipwreck').length})</div>
                <div className="legend-item"><span className="dot plane"></span> Aircraft ({detections.filter(d=>d.classification==='aircraft').length})</div>
              </div>
            )}
          </div>
        ) : (
          <div className="viewer-empty-state">
            <Layers size={48} className="empty-icon" />
            <p>Select a benchmark sonar swath above or upload a raw file to inspect acoustic imagery.</p>
          </div>
        )}
      </div>
    </div>
  );
}
