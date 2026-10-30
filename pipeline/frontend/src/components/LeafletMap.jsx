import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { MapPin, Navigation } from 'lucide-react';

export default function LeafletMap({ telemetryTrack, detections, onSelectAnomaly }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersLayerRef = useRef(null);
  const trackLayerRef = useRef(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      // Initialize map centered at survey location
      const map = L.map(mapContainerRef.current, {
        center: [13.0827, 80.2707],
        zoom: 14,
        zoomControl: true,
        attributionControl: false
      });

      // Dark Matter GIS Nautical Tiles
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        subdomains: 'abcd',
      }).addTo(map);

      trackLayerRef.current = L.layerGroup().addTo(map);
      markersLayerRef.current = L.layerGroup().addTo(map);

      mapInstanceRef.current = map;
    }

    const map = mapInstanceRef.current;
    const trackLayer = trackLayerRef.current;
    const markersLayer = markersLayerRef.current;

    trackLayer.clearLayers();
    markersLayer.clearLayers();

    // 1. Draw AUV Trackline
    const trackCoords = telemetryTrack?.track_coords || [];
    if (trackCoords.length > 0) {
      const polyline = L.polyline(trackCoords, {
        color: '#00e5ff',
        weight: 3,
        opacity: 0.85,
        dashArray: '4, 8'
      }).addTo(trackLayer);

      // Start marker (Towfish deployment)
      L.circleMarker(trackCoords[0], {
        radius: 6,
        color: '#00e676',
        fillColor: '#00e676',
        fillOpacity: 1
      }).bindTooltip('AUV Deployment Start').addTo(trackLayer);

      // Current Towfish position
      L.circleMarker(trackCoords[trackCoords.length - 1], {
        radius: 7,
        color: '#00e5ff',
        fillColor: '#00e5ff',
        fillOpacity: 0.9
      }).bindTooltip('Current Towfish Position').addTo(trackLayer);

      map.fitBounds(polyline.getBounds(), { padding: [35, 35], maxZoom: 17 });
    }

    // 2. Draw Anomaly Hazard Markers
    if (detections && detections.length > 0) {
      detections.forEach((d) => {
        const lat = d.latitude;
        const lon = d.longitude;
        if (!lat || !lon) return;

        const isNet = d.classification === 'ghost_net';
        const isWreck = d.classification === 'shipwreck';
        const markerColor = isNet ? '#00e5ff' : isWreck ? '#ff3366' : '#ffb703';

        const marker = L.circleMarker([lat, lon], {
          radius: isNet ? 9 : 8,
          color: '#ffffff',
          weight: 2,
          fillColor: markerColor,
          fillOpacity: 0.9
        }).addTo(markersLayer);

        const popupContent = `
          <div style="font-family: 'Outfit', sans-serif; color: #0a1120; min-width: 200px;">
            <div style="font-weight: 800; font-size: 14px; color: ${isNet ? '#0077b6' : '#d00000'}; margin-bottom: 4px;">
              ${d.anomaly_id}: ${d.classification.toUpperCase()}
            </div>
            <div style="font-size: 12px; margin-bottom: 2px;">
              <strong>Composite Score:</strong> ${(d.composite_score_pct ?? d.calibrated_confidence_pct ?? 0).toFixed(1)}/100 <span style="font-size: 10px; color: #64748b;">(${d.review_status || d.confidence_tier || 'Review'})</span>
            </div>
            <div style="font-size: 11px; margin-bottom: 2px; color: #475569;">
              <strong>Shadow:</strong> ${d.physics_verified ? 'PASS' : 'FAIL'} | <strong>Quality:</strong> ${d.georeference_quality || 'unnavigated'}
            </div>
            <div style="font-size: 12px; margin-bottom: 2px;">
              <strong>Coordinates:</strong> ${lat.toFixed(6)}° N, ${lon.toFixed(6)}° E (${d.gps_source || 'GPS'})
            </div>
            <div style="font-size: 12px; margin-bottom: 2px;">
              <strong>Dimensions:</strong> ${d.estimated_length_m}m × ${d.estimated_width_m}m
            </div>
            <div style="font-size: 12px; margin-bottom: 6px;">
              <strong>Seabed Height:</strong> ${d.estimated_height_m}m (${d.channel.toUpperCase()})
            </div>
            <div style="font-size: 11px; padding: 4px 6px; background: #e0f2fe; border-radius: 4px; color: #0369a1;">
              <strong>Action:</strong> ${d.recommended_action}
            </div>
          </div>
        `;

        marker.bindPopup(popupContent);
        marker.on('click', () => {
          if (onSelectAnomaly) onSelectAnomaly(d);
        });
      });
    }
  }, [telemetryTrack, detections, onSelectAnomaly]);

  return (
    <div className="map-card-container">
      <div className="map-header-strip">
        <div className="map-title-row">
          <Navigation size={16} className="cyan" />
          <h3 className="map-title">GEOSPATIAL SURVEY & HAZARD PINPOINT MAP</h3>
        </div>
        <div className="map-legend-row">
          <span className="legend-chip"><span className="legend-dot a-net"></span> Ghost Net</span>
          <span className="legend-chip"><span className="legend-dot a-wreck"></span> Shipwreck</span>
          <span className="legend-chip"><span className="legend-dot a-other"></span> Aircraft / Debris</span>
          <span className="legend-chip"><span className="legend-line"></span> Survey Track</span>
        </div>
      </div>
      <div ref={mapContainerRef} className="leaflet-map-element" style={{ height: '320px', width: '100%' }}></div>
    </div>
  );
}
