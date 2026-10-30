import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

// Status badge config
const STATUS_CONFIG = {
  pending:       { label: 'Pending',                color: '#ffb703', bg: 'rgba(255,183,3,0.15)',   icon: '⏳' },
  uploading:     { label: 'Uploading',              color: '#00b4d8', bg: 'rgba(0,180,216,0.15)',   icon: '⬆' },
  uploaded:      { label: 'Uploaded',               color: '#00e676', bg: 'rgba(0,230,118,0.15)',   icon: '✓' },
  failed:        { label: 'Failed (retrying)',       color: '#ff3366', bg: 'rgba(255,51,102,0.15)',  icon: '⟳' },
  failed_manual: { label: 'Failed – manual retry',  color: '#ff3366', bg: 'rgba(255,51,102,0.25)',  icon: '✕' },
};

export default function CloudSyncPanel({ hasResults }) {
  const [cloudData, setCloudData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [actionLoading, setActionLoading] = useState({}); // per-entry loading state

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/cloud/status`);
      if (res.ok) {
        const data = await res.json();
        setCloudData(data);
      }
    } catch (err) {
      console.error('Cloud status fetch error:', err);
    }
  }, []);

  // Fetch on mount and on expand
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Auto-refresh every 30 seconds when expanded
  useEffect(() => {
    if (!isExpanded) return;
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [isExpanded, fetchStatus]);

  const handleUploadSingle = async (format) => {
    setActionLoading(prev => ({ ...prev, [`upload_${format}`]: true }));
    try {
      const res = await fetch(`${API_BASE}/api/cloud/upload?report_format=${format}`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        console.error('Upload error:', err);
      }
      await fetchStatus();
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setActionLoading(prev => ({ ...prev, [`upload_${format}`]: false }));
    }
  };

  const handleUploadAll = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/cloud/upload-all`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        console.error('Upload all error:', err);
      }
      await fetchStatus();
    } catch (err) {
      console.error('Upload all failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = async (entryId) => {
    setActionLoading(prev => ({ ...prev, [`retry_${entryId}`]: true }));
    try {
      const res = await fetch(`${API_BASE}/api/cloud/retry/${entryId}`, { method: 'POST' });
      await fetchStatus();
    } catch (err) {
      console.error('Retry failed:', err);
    } finally {
      setActionLoading(prev => ({ ...prev, [`retry_${entryId}`]: false }));
    }
  };

  const handleRetryAll = async () => {
    setIsLoading(true);
    try {
      await fetch(`${API_BASE}/api/cloud/retry-all`, { method: 'POST' });
      await fetchStatus();
    } catch (err) {
      console.error('Retry all failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCheckConnectivity = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/cloud/connectivity`);
      await fetchStatus();
    } catch (err) {
      console.error('Connectivity check failed:', err);
    }
  };

  const engineStatus = cloudData?.engine_status || {};
  const entries = cloudData?.entries || [];
  const isOnline = engineStatus.is_online || false;
  const isEnabled = cloudData?.enabled !== false;
  const stats = engineStatus.manifest_stats || {};
  const byStatus = stats.by_status || {};

  const pendingCount = (byStatus.pending || 0) + (byStatus.failed || 0);
  const failedManualCount = byStatus.failed_manual || 0;
  const uploadedCount = byStatus.uploaded || 0;

  // Upload buttons for each report format
  const reportFormats = [
    { key: 'csv',     label: 'CSV',     icon: '📊' },
    { key: 'json',    label: 'JSON',    icon: '📋' },
    { key: 'geojson', label: 'GeoJSON', icon: '🗺️' },
    { key: 'pdf',     label: 'PDF',     icon: '📄' },
  ];

  return (
    <div className="cloud-sync-panel">
      {/* Collapsed header bar */}
      <div className="cloud-sync-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="cloud-sync-title-group">
          <span className="cloud-sync-icon">☁</span>
          <span className="cloud-sync-title">CLOUD SYNC</span>
          <span className={`cloud-connectivity-dot ${isOnline ? 'online' : 'offline'}`} />
          <span className="cloud-connectivity-label">
            {isOnline ? 'Connected' : 'Offline'}
          </span>
        </div>

        <div className="cloud-sync-summary-badges">
          {pendingCount > 0 && (
            <span className="cloud-summary-badge pending">
              {pendingCount} pending
            </span>
          )}
          {failedManualCount > 0 && (
            <span className="cloud-summary-badge failed">
              {failedManualCount} failed
            </span>
          )}
          {uploadedCount > 0 && (
            <span className="cloud-summary-badge uploaded">
              {uploadedCount} uploaded
            </span>
          )}
          {stats.total_entries === 0 && (
            <span className="cloud-summary-badge empty">No reports queued</span>
          )}
        </div>

        <button className="cloud-expand-btn" onClick={(e) => { e.stopPropagation(); setIsExpanded(!isExpanded); }}>
          {isExpanded ? '▲' : '▼'}
        </button>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="cloud-sync-body">
          {/* Action buttons row */}
          <div className="cloud-actions-row">
            <div className="cloud-upload-buttons">
              <span className="cloud-actions-label">Upload to Cloud:</span>
              {reportFormats.map(fmt => (
                <button
                  key={fmt.key}
                  className={`btn-cloud-upload ${fmt.key}`}
                  disabled={!hasResults || actionLoading[`upload_${fmt.key}`]}
                  onClick={() => handleUploadSingle(fmt.key)}
                  title={`Upload ${fmt.label} report to Azure`}
                >
                  {actionLoading[`upload_${fmt.key}`] ? (
                    <span className="spin-icon">⟳</span>
                  ) : (
                    <span>{fmt.icon}</span>
                  )}
                  {fmt.label}
                </button>
              ))}
            </div>

            <div className="cloud-bulk-buttons">
              <button
                className="btn-cloud-bulk upload-all"
                onClick={handleUploadAll}
                disabled={isLoading || pendingCount === 0}
                title="ZIP all pending reports and upload"
              >
                {isLoading ? <span className="spin-icon">⟳</span> : '📦'}
                Upload All (ZIP)
              </button>

              <button
                className="btn-cloud-bulk retry-all"
                onClick={handleRetryAll}
                disabled={isLoading || (pendingCount === 0 && failedManualCount === 0)}
              >
                ⟳ Retry All
              </button>

              <button className="btn-cloud-bulk refresh" onClick={fetchStatus}>
                🔄 Refresh
              </button>

              <button className="btn-cloud-bulk connectivity" onClick={handleCheckConnectivity}>
                📡 Check Link
              </button>
            </div>
          </div>

          {/* Engine info strip */}
          {!isEnabled && (
            <div className="cloud-info-banner warning">
              ⚠ Cloud sync module not loaded. Install <code>azure-storage-blob</code> and set <code>AZURE_STORAGE_CONNECTION_STRING</code> env var.
            </div>
          )}
          {isEnabled && !engineStatus.azure_configured && (
            <div className="cloud-info-banner info">
              ℹ Azure not configured. Set <code>AZURE_STORAGE_CONNECTION_STRING</code> environment variable to enable uploads.
              Reports are queued locally and will upload when configured.
            </div>
          )}

          {/* Manifest entries table */}
          {entries.length > 0 ? (
            <div className="cloud-table-wrap">
              <table className="cloud-manifest-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>FILE</th>
                    <th>FORMAT</th>
                    <th>SIZE</th>
                    <th>STATUS</th>
                    <th>RETRIES</th>
                    <th>LAST ATTEMPT</th>
                    <th>ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map(entry => {
                    const statusCfg = STATUS_CONFIG[entry.upload_status] || STATUS_CONFIG.pending;
                    const canRetry = entry.upload_status === 'failed' || entry.upload_status === 'failed_manual';
                    const isEntryLoading = actionLoading[`retry_${entry.id}`];

                    return (
                      <tr key={entry.id} className="cloud-entry-row">
                        <td className="cloud-cell-id font-mono">{entry.id}</td>
                        <td className="cloud-cell-file" title={entry.file_path}>
                          {entry.file_name}
                        </td>
                        <td className="cloud-cell-format">
                          <span className={`format-tag ${entry.report_format || 'other'}`}>
                            {(entry.report_format || '—').toUpperCase()}
                          </span>
                        </td>
                        <td className="cloud-cell-size font-mono">
                          {entry.file_size_bytes > 1024
                            ? `${(entry.file_size_bytes / 1024).toFixed(1)} KB`
                            : `${entry.file_size_bytes} B`}
                        </td>
                        <td>
                          <span
                            className="cloud-status-badge"
                            style={{ color: statusCfg.color, background: statusCfg.bg }}
                          >
                            {statusCfg.icon} {statusCfg.label}
                          </span>
                        </td>
                        <td className="cloud-cell-retries font-mono">
                          {entry.retry_count}/{entry.max_retries}
                        </td>
                        <td className="cloud-cell-ts font-mono">
                          {entry.last_attempt_ts
                            ? new Date(entry.last_attempt_ts).toLocaleTimeString()
                            : '—'}
                        </td>
                        <td>
                          {canRetry && (
                            <button
                              className="btn-cloud-retry"
                              onClick={() => handleRetry(entry.id)}
                              disabled={isEntryLoading}
                            >
                              {isEntryLoading ? '...' : '⟳ Retry Now'}
                            </button>
                          )}
                          {entry.upload_status === 'uploaded' && entry.azure_blob_url && (
                            <span className="cloud-uploaded-check" title={entry.azure_blob_url}>
                              ✓ Verified
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="cloud-empty-state">
              <span className="cloud-empty-icon">☁</span>
              <span>No reports queued for cloud upload yet. Export a report to begin.</span>
            </div>
          )}

          {/* Error messages for failed_manual entries */}
          {entries.filter(e => e.upload_status === 'failed_manual' && e.error_message).length > 0 && (
            <div className="cloud-error-summary">
              <span className="cloud-error-title">⚠ Reports requiring manual retry:</span>
              {entries.filter(e => e.upload_status === 'failed_manual').map(e => (
                <div key={e.id} className="cloud-error-item">
                  <strong>{e.file_name}</strong>: {e.error_message}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
