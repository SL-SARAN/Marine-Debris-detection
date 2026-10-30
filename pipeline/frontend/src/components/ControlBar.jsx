import React, { useRef } from 'react';
import { Play, Upload, Download, FileSpreadsheet, FileCode, FileText, Sliders, RefreshCw } from 'lucide-react';

export default function ControlBar({
  samples,
  selectedSample,
  onSelectSample,
  onUploadFile,
  uploadedFileName,
  useWavelet,
  setUseWavelet,
  useLee,
  setUseLee,
  confThresh,
  setConfThresh,
  onRunPipeline,
  isLoading,
  onExportCsv,
  onExportJson,
  onExportPdf,
  hasResults
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUploadFile(e.target.files[0]);
    }
  };

  return (
    <section className="control-bar-panel">
      <div className="control-left-section">
        {/* Sample Selection */}
        <div className="control-item">
          <label className="control-label">BENCHMARK SONAR SWATH:</label>
          <select 
            className="control-select"
            value={selectedSample}
            onChange={(e) => onSelectSample(e.target.value)}
            disabled={isLoading}
          >
            <option value="">-- Choose Sample Swath --</option>
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.type === 'xtf' ? '🌊 [XTF LOG] ' : '🎯 [SWATH] '} {s.name}
              </option>
            ))}
          </select>
        </div>

        {/* Upload Custom Sonar */}
        <div className="control-item">
          <label className="control-label">OR UPLOAD RAW LOG / IMAGE:</label>
          <div className="upload-btn-wrap">
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept=".png,.jpg,.jpeg,.tif,.tiff,.xtf"
              onChange={handleFileChange} 
            />
            <button 
              className={`btn-upload ${uploadedFileName ? 'has-file' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
            >
              <Upload size={16} />
              <span>{uploadedFileName ? uploadedFileName.substring(0, 18) + '...' : 'Upload Sonar Image / .XTF'}</span>
            </button>
          </div>
        </div>

        {/* Filter Controls */}
        <div className="control-item filters-group">
          <label className="control-label"><Sliders size={14} /> ACOUSTIC FILTERS:</label>
          <div className="filter-checkboxes">
            <label className="filter-toggle">
              <input 
                type="checkbox" 
                checked={useWavelet} 
                onChange={(e) => setUseWavelet(e.target.checked)} 
              />
              <span>2D Wavelet DWT</span>
            </label>
            <label className="filter-toggle">
              <input 
                type="checkbox" 
                checked={useLee} 
                onChange={(e) => setUseLee(e.target.checked)} 
              />
              <span>Enhanced Lee</span>
            </label>
          </div>
        </div>

        {/* Confidence Threshold */}
        <div className="control-item conf-slider-group">
          <div className="slider-label-row">
            <label className="control-label">CONFIDENCE THRESHOLD:</label>
            <span className="slider-value">{(confThresh * 100).toFixed(0)}%</span>
          </div>
          <input 
            type="range" 
            min="0.10" 
            max="0.80" 
            step="0.05"
            value={confThresh} 
            onChange={(e) => setConfThresh(parseFloat(e.target.value))}
            className="control-range"
          />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="control-right-section">
        <button 
          className={`btn-primary-run ${isLoading ? 'loading' : ''}`}
          onClick={onRunPipeline}
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <RefreshCw size={18} className="spin-icon" />
              <span>PROCESSING SWATH...</span>
            </>
          ) : (
            <>
              <Play size={18} fill="currentColor" />
              <span>RUN CV PIPELINE</span>
            </>
          )}
        </button>

        {hasResults && (
          <div className="export-btn-group">
            <button className="btn-export csv" onClick={onExportCsv} title="Download CSV Anomaly Report">
              <FileSpreadsheet size={16} />
              <span>CSV</span>
            </button>
            <button className="btn-export json" onClick={onExportJson} title="Download JSON Survey Payload">
              <FileCode size={16} />
              <span>JSON</span>
            </button>
            <button className="btn-export pdf" onClick={onExportPdf} title="Download Printable Maritime PDF Audit Report">
              <FileText size={16} />
              <span>PDF AUDIT</span>
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
