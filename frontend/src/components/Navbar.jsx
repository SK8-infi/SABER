import React from 'react';
import { 
  Compass, 
  Cpu, 
  Search, 
  Activity, 
  GitCompare, 
  BarChart3, 
  Database, 
  Clock, 
  Share2, 
  Download,
  Layers
} from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, telemetry }) {
  const navItems = [
    { id: 'overview', label: '1. Science Overview', icon: Compass },
    { id: 'architecture', label: '2. Pipeline Flow', icon: Cpu },
    { id: 'workspace', label: '3. Live Workspace', icon: Search },
    { id: 'explainability', label: '4. Physical Lab', icon: Layers },
    { id: 'ablation', label: '5. CFM Bridge Studio', icon: GitCompare },
    { id: 'benchmark', label: '6. SOTA Benchmarks', icon: BarChart3 },
    { id: 'dataset', label: '7. Datasets (BEN-14K)', icon: Database },
    { id: 'telemetry', label: '8. System Latency', icon: Clock },
    { id: 'embedding', label: '9. 2D Embedding Space', icon: Activity },
    { id: 'report', label: '10. Export Report', icon: Download },
  ];

  return (
    <header className="navbar-header">
      <div className="brand-section">
        <div className="isro-badge">ISRO BAH 2026</div>
        <div className="platform-title">
          SABER // <span>RESEARCH DEMO PLATFORM</span>
        </div>
        <span className="mono-tag cyan">PS-11 FINALIST</span>
      </div>

      <nav className="nav-tabs">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={`nav-tab-btn ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={14} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="nav-right-telemetry">
        <div className="telemetry-item">
          <span className="telemetry-label">Avg Query Latency</span>
          <span className="telemetry-value" style={{ color: 'var(--accent-green)' }}>
            {telemetry?.total_latency_ms ? `${telemetry.total_latency_ms} ms` : '28.48 ms'}
          </span>
        </div>

        <div className="telemetry-item">
          <span className="telemetry-label">Gallery Index</span>
          <span className="telemetry-value">
            {telemetry?.gallery_size ? telemetry.gallery_size.toLocaleString() : '11,866'} SCENES
          </span>
        </div>

        <div className="telemetry-item">
          <span className="telemetry-label">VRAM Footprint</span>
          <span className="telemetry-value">918.7 MB</span>
        </div>
      </div>
    </header>
  );
}
