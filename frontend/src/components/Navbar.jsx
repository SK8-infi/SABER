import { useState } from 'react';
import { Menu, X } from 'lucide-react';

export default function Navbar({ tabs, active, setTab, telemetry, onMenuToggle, sidebarOpen }) {
  return (
    <header className="navbar">
      {/* Mobile sidebar toggle */}
      <button
        className="sidebar-toggle btn btn--ghost btn--sm"
        onClick={onMenuToggle}
        title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
      >
        {sidebarOpen ? <X size={16} /> : <Menu size={16} />}
      </button>

      <div className="navbar-brand">
        <span className="brand-badge">ISRO BAH 2026</span>
        <span className="brand-name">SABER <span>// RESEARCH</span></span>
        <span className="tag tag--cyan" style={{ fontSize: '0.6rem' }}>PS-11</span>
      </div>

      <nav className="navbar-tabs">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`tab-btn${active === t.id ? ' active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.id === 'demo' && (
              <span className="live-dot" title="Live API" />
            )}
            {t.label}
          </button>
        ))}
      </nav>

      <div className="navbar-stats">
        <div className="stat-item">
          <span className="stat-label">Latency</span>
          <span
            className="stat-value"
            style={{
              color: telemetry?.total_latency_ms < 30
                ? 'var(--green)'
                : telemetry?.total_latency_ms
                ? 'var(--saffron)'
                : 'var(--cyan)',
              transition: 'color 0.3s',
            }}
          >
            {telemetry?.total_latency_ms
              ? `${telemetry.total_latency_ms} ms`
              : '— ms'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Gallery</span>
          <span className="stat-value">
            {telemetry?.gallery_size
              ? telemetry.gallery_size.toLocaleString()
              : '11,866'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">VRAM</span>
          <span className="stat-value">
            {telemetry?.vram_allocated_mb
              ? `${telemetry.vram_allocated_mb} MB`
              : '918.7 MB'}
          </span>
        </div>
      </div>
    </header>
  );
}
