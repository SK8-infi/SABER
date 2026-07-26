export default function Navbar({ tabs, active, setTab, telemetry }) {
  return (
    <header className="navbar">
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
          <span className="stat-value">
            {telemetry?.total_latency_ms ? `${telemetry.total_latency_ms} ms` : '28.48 ms'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Gallery</span>
          <span className="stat-value">
            {telemetry?.gallery_size ? telemetry.gallery_size.toLocaleString() : '11,866'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">VRAM</span>
          <span className="stat-value">
            {telemetry?.vram_allocated_mb ? `${telemetry.vram_allocated_mb} MB` : '918.7 MB'}
          </span>
        </div>
      </div>
    </header>
  );
}
