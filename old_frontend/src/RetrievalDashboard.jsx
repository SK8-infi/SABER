import React, { useState } from 'react';
import './index.css';
import { 
  Activity, 
  Map as MapIcon,
  ChevronRight,
  Database,
  Layers,
  Search,
  UploadCloud,
  Crosshair,
  BarChart2,
  Clock,
  Target
} from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { BarChart, Bar, XAxis, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine, PieChart, Pie, Cell, Legend } from 'recharts';

import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

// MOCK DATA
const similarityData = [
  { name: '70', count: 12 }, { name: '75', count: 28 }, { name: '80', count: 85 },
  { name: '85', count: 190 }, { name: '90', count: 410 }, { name: '95', count: 235 }, 
  { name: '98', count: 95 }, { name: '100', count: 15 },
];

const modalityData = [
  { name: 'SAR', value: 40, color: '#a1a1aa' },
  { name: 'Multispectral', value: 40, color: '#71717a' },
  { name: 'Panchromatic', value: 10, color: '#52525b' },
  { name: 'RGB', value: 10, color: '#3f3f46' },
];

const initialResults = [
  { id: 1, type: 'SAR', score: '96.8', date: '2024-08-13 05:42', location: '23.45°, 88.31°', res: '10m', img: '/img_res1.png', selected: true },
  { id: 2, type: 'MS', score: '95.7', date: '2024-08-12 05:30', location: '23.47°, 88.28°', res: '10m', img: '/img_res2.png', selected: false },
  { id: 3, type: 'PAN', score: '95.2', date: '2024-08-13 05:35', location: '23.44°, 88.30°', res: '2.5m', img: '/img_res3.png', selected: true },
  { id: 4, type: 'SAR', score: '94.1', date: '2024-08-13 05:44', location: '23.41°, 88.35°', res: '10m', img: '/img_res4.png', selected: false },
  { id: 5, type: 'MS', score: '93.8', date: '2024-08-10 05:28', location: '23.48°, 88.33°', res: '10m', img: '/img_res5.png', selected: false },
  { id: 6, type: 'RGB', score: '92.1', date: '2024-08-09 11:20', location: '23.50°, 88.10°', res: '5m', img: '/img_res1.png', selected: false },
];

export default function RetrievalDashboard() {
  const [results, setResults] = useState(initialResults);

  const toggleSelect = (id) => {
    setResults(results.map(r => r.id === id ? { ...r, selected: !r.selected } : r));
  };

  return (
    <div className="dashboard-container">
      
      {/* MINIMALIST HEADER */}
      <header className="dashboard-header">
        <div className="header-left">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-primary)', fontWeight: '700', fontSize: '1.25rem', letterSpacing: '-0.02em' }}>
            <img src="/logo.png" alt="SABER Logo" style={{ height: '36px', width: '36px', objectFit: 'contain' }} />
            <span>SABER // PRECISION RETRIEVAL</span>
          </div>
        </div>
        
        <div className="header-stats">
          <button className="compare-btn">
            <Layers size={16} /> COMPARE ({results.filter(r => r.selected).length})
          </button>

          <div className="stat-item">
            <span className="stat-label">System Status</span>
            <span className="stat-value"><span className="status-dot active"></span> OPERATIONAL</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Gallery Index</span>
            <span className="stat-value">12,467,392 <span style={{ color: 'var(--text-tertiary)' }}>SCENES</span></span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Engine</span>
            <span className="stat-value">FAISS IVF-PQ</span>
          </div>
        </div>
      </header>

      {/* GRID LAYOUT */}
      <div className="dashboard-grid">
        
        {/* LEFT: QUERY MODULE */}
        <div className="panel">
          <div className="panel-header">
            <span>Query Parameters</span>
            <Search size={14} color="var(--text-tertiary)" />
          </div>
          <div className="panel-content">
            
            <div className="image-dropzone">
              <img src="/img_query.png" alt="Query" style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8 }} />
              <div className="dropzone-overlay">
                <UploadCloud size={24} />
                <span>Change Image</span>
              </div>
              <div style={{ position: 'absolute', bottom: '12px', left: '12px', fontSize: '0.6875rem', fontFamily: 'JetBrains Mono', color: 'var(--text-primary)', textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}>
                SAR_FLOOD_0813.TIFF
              </div>
            </div>

            <div className="input-group">
              <label>Target Modality</label>
              <select className="elegant-select" defaultValue="any">
                <option value="any">Cross-Modal (Any Sensor)</option>
                <option value="sar">SAR Only</option>
                <option value="optical">Optical Only</option>
              </select>
            </div>

            <div className="input-group">
              <label>Top-K Candidates</label>
              <select className="elegant-select" defaultValue="10">
                <option value="5">5 Candidates</option>
                <option value="10">10 Candidates</option>
                <option value="50">50 Candidates</option>
              </select>
            </div>

            <div className="input-group">
              <label>Temporal Range</label>
              <input type="text" className="elegant-input" defaultValue="2020-01-01 / 2025-05-15" />
            </div>

            <div style={{ marginTop: 'auto', paddingTop: '32px' }}>
              <button className="primary-btn">
                <Crosshair size={16} /> EXECUTE RETRIEVAL
              </button>
            </div>
          </div>
        </div>

        {/* CENTER: RESULTS & ANALYTICS */}
        <div className="main-content">
          
          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">
              <span>Retrieval Candidates</span>
              <span className="data-mono" style={{ color: 'var(--text-secondary)' }}>N={results.length}</span>
            </div>
            <div className="panel-content" style={{ padding: '24px' }}>
              <div className="results-grid">
                {results.map(res => (
                  <div className={`result-card ${res.selected ? 'selected' : ''}`} key={res.id}>
                    <input type="checkbox" className="elegant-checkbox" checked={res.selected} onChange={() => toggleSelect(res.id)} />
                    <div className="card-thumb">
                      <img src={res.img} alt={`Result ${res.id}`} />
                      <div className="card-overlay">
                        <span>VIEW DETAILS</span>
                      </div>
                    </div>
                    <div className="card-meta">
                      <div className="card-meta-row">
                        <span style={{ color: 'var(--text-primary)' }}>{res.type}</span>
                        <span style={{ color: res.selected ? 'var(--accent-saffron)' : 'var(--text-primary)', fontWeight: 500 }}>{res.score}%</span>
                      </div>
                      <div className="card-meta-row">
                        <span>{res.res}</span>
                        <span>{res.date}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Analytics Row */}
          <div className="metrics-row" style={{ gridTemplateColumns: '1.5fr 1fr' }}>
            <div className="panel" style={{ height: '240px' }}>
              <div className="panel-header">
                <span>Similarity Distribution</span>
                <BarChart2 size={14} color="var(--text-tertiary)" />
              </div>
              <div className="panel-content" style={{ padding: '16px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={similarityData}>
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#71717a', fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                    <RechartsTooltip 
                      cursor={{ fill: '#18181b' }} 
                      contentStyle={{ backgroundColor: '#0a0a0b', border: '1px solid #27272a', fontFamily: 'JetBrains Mono', fontSize: '12px' }} 
                      itemStyle={{ color: '#e58d57' }}
                    />
                    <Bar dataKey="count" fill="#3f3f46" radius={[2, 2, 0, 0]} />
                    <ReferenceLine x="90" stroke="#e58d57" strokeDasharray="3 3" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel" style={{ height: '240px' }}>
              <div className="panel-header">
                <span>Modality Mix</span>
                <Target size={14} color="var(--text-tertiary)" />
              </div>
              <div className="panel-content" style={{ padding: '16px 16px 16px 0', display: 'flex', alignItems: 'center' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={modalityData} innerRadius={50} outerRadius={70} paddingAngle={2} dataKey="value" stroke="none">
                      {modalityData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Legend verticalAlign="middle" align="right" layout="vertical" iconType="circle" wrapperStyle={{ fontSize: '10px', fontFamily: 'JetBrains Mono' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

        </div>

        {/* RIGHT: SPATIAL & TELEMETRY */}
        <div className="main-content">
          
          <div className="panel" style={{ height: '360px' }}>
            <div className="panel-header">
              <span>Spatial Overview</span>
              <MapIcon size={14} color="var(--text-tertiary)" />
            </div>
            <div className="panel-content" style={{ padding: 0, position: 'relative' }}>
              <MapContainer center={[23.45, 88.31]} zoom={5} style={{ height: '100%', width: '100%', filter: 'grayscale(100%) invert(100%) hue-rotate(180deg)' }} zoomControl={false}>
                <TileLayer
                  url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                  attribution=''
                />
                <Marker position={[23.45, 88.31]} />
              </MapContainer>
              
              <div className="map-radar"></div>

              <div style={{ position: 'absolute', bottom: '16px', left: '16px', zIndex: 1000, backgroundColor: '#0a0a0b', border: '1px solid #27272a', padding: '8px 12px', fontSize: '0.6875rem', fontFamily: 'JetBrains Mono', color: 'var(--text-secondary)' }}>
                LAT 23.45° / LON 88.31°
              </div>
            </div>
          </div>

          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">
              <span>Session Telemetry</span>
              <Clock size={14} color="var(--text-tertiary)" />
            </div>
            <div className="panel-content" style={{ padding: '24px 32px' }}>
              <table className="data-table">
                <tbody>
                  <tr><td>Query Modality</td><td>SAR</td></tr>
                  <tr><td>Query Latency</td><td><span style={{ color: 'var(--text-primary)' }}>0.82 ms</span></td></tr>
                  <tr><td>ANN Search Time</td><td>0.41 ms</td></tr>
                  <tr><td>Re-ranking Time</td><td>0.29 ms</td></tr>
                  <tr><td>Cross-modal F1@10</td><td>0.78</td></tr>
                  <tr><td>mAP</td><td>0.65</td></tr>
                  <tr style={{ borderTop: '1px solid var(--border-strong)' }}>
                    <td style={{ paddingTop: '16px' }}>Session Hash</td>
                    <td style={{ paddingTop: '16px', color: 'var(--text-secondary)' }}>9f3a7c2e</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          
        </div>

      </div>

    </div>
  );
}

// export default RetrievalDashboard;
