import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import KPICards from './components/KPICards';
import ControlBar from './components/ControlBar';
import PipelineViewer from './components/PipelineViewer';
import LeafletMap from './components/LeafletMap';
import AnomalyTable from './components/AnomalyTable';
import TelemetryDrawer from './components/TelemetryDrawer';
import CloudSyncPanel from './components/CloudSyncPanel';
import './App.css';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [healthData, setHealthData] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [useWavelet, setUseWavelet] = useState(true);
  const [useLee, setUseLee] = useState(true);
  const [confThresh, setConfThresh] = useState(0.20);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [selectedAnomalyId, setSelectedAnomalyId] = useState(null);

  // Fetch health & sample list on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => setHealthData(data))
      .catch((err) => console.error('Health check failed:', err));

    fetch(`${API_BASE}/api/samples`)
      .then((res) => res.json())
      .then((data) => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
          // Pre-select first ghost net sample
          const defaultSample = data.samples[0].id;
          setSelectedSample(defaultSample);
          runPipelineWithSample(defaultSample, true, true, 0.20);
        }
      })
      .catch((err) => console.error('Failed to fetch samples:', err));
  }, []);

  const runPipelineWithSample = async (sampleId, wavelet = useWavelet, lee = useLee, conf = confThresh) => {
    if (!sampleId) return;
    setIsLoading(true);
    try {
      const url = `${API_BASE}/api/process_sonar?sample_id=${encodeURIComponent(sampleId)}&use_wavelet=${wavelet}&use_lee=${lee}&conf_threshold=${conf}`;
      const res = await fetch(url, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error('Pipeline processing failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunPipeline = async () => {
    if (uploadedFile) {
      setIsLoading(true);
      try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        const url = `${API_BASE}/api/process_sonar?use_wavelet=${useWavelet}&use_lee=${useLee}&conf_threshold=${confThresh}`;
        const res = await fetch(url, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        const data = await res.json();
        setResults(data);
      } catch (err) {
        console.error('Uploaded swath processing failed:', err);
      } finally {
        setIsLoading(false);
      }
    } else if (selectedSample) {
      runPipelineWithSample(selectedSample, useWavelet, useLee, confThresh);
    }
  };

  const handleUploadFile = (file) => {
    setUploadedFile(file);
    setSelectedSample('');
  };

  const handleExportCsv = () => {
    window.open(`${API_BASE}/api/export/csv`, '_blank');
  };

  const handleExportJson = () => {
    window.open(`${API_BASE}/api/export/json`, '_blank');
  };

  const handleExportPdf = () => {
    window.open(`${API_BASE}/api/export/pdf`, '_blank');
  };

  return (
    <div className="app-layout">
      {/* Top Header */}
      <Navbar healthData={healthData} latencyData={results?.latency_benchmark} />

      <main className="app-main-content">
        {/* KPI Metric Summary Strip */}
        <KPICards 
          summary={results?.summary} 
          latency={results?.latency_benchmark} 
        />

        {/* Control and Ingestion Bar */}
        <ControlBar
          samples={samples}
          selectedSample={selectedSample}
          onSelectSample={(s) => {
            setSelectedSample(s);
            setUploadedFile(null);
            runPipelineWithSample(s, useWavelet, useLee, confThresh);
          }}
          onUploadFile={handleUploadFile}
          uploadedFileName={uploadedFile?.name}
          useWavelet={useWavelet}
          setUseWavelet={setUseWavelet}
          useLee={useLee}
          setUseLee={setUseLee}
          confThresh={confThresh}
          setConfThresh={setConfThresh}
          onRunPipeline={handleRunPipeline}
          isLoading={isLoading}
          onExportCsv={handleExportCsv}
          onExportJson={handleExportJson}
          onExportPdf={handleExportPdf}
          hasResults={Boolean(results && results.detections && results.detections.length > 0)}
        />

        {/* Primary 2-Column Content Area */}
        <div className="content-columns-grid">
          {/* Left Column: 4-Stage Acoustic Waterfall Inspector */}
          <section className="left-pipeline-col">
            <PipelineViewer
              stages={results?.stages}
              speckleMetrics={results?.speckle_metrics}
              imageShape={results?.image_shape}
              detections={results?.detections}
              isLoading={isLoading}
            />
          </section>

          {/* Right Column: GIS Map and Structured Table */}
          <section className="right-gis-col">
            {/* GIS Map with Live Track & Pins */}
            <LeafletMap
              telemetryTrack={results?.telemetry_track}
              detections={results?.detections}
              onSelectAnomaly={(d) => setSelectedAnomalyId(d.anomaly_id)}
            />

            {/* Anomaly Inspection Table */}
            <AnomalyTable
              detections={results?.detections}
              selectedAnomalyId={selectedAnomalyId}
              onSelectAnomaly={(d) => setSelectedAnomalyId(d.anomaly_id)}
            />
          </section>
        </div>

        {/* Bottom Hardware & Edge Latency Benchmark Telemetry */}
        <TelemetryDrawer 
          latency={results?.latency_benchmark} 
          speckleMetrics={results?.speckle_metrics}
          totalAnomalies={results?.detections?.length || 0}
        />

        {/* Cloud Sync Panel — optional, non-blocking */}
        <CloudSyncPanel
          hasResults={Boolean(results && results.detections && results.detections.length > 0)}
        />
      </main>
    </div>
  );
}
