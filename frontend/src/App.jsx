import { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import RetrievalWorkspace from './components/RetrievalWorkspace';
import ResultsDashboard from './components/ResultsDashboard';
import AboutPanel from './components/AboutPanel';
import CompareModal from './components/CompareModal';
import GalleryBrowser from './components/GalleryBrowser';
import EmbeddingVisualizer from './components/EmbeddingVisualizer';

const TABS = [
  { id: 'demo',      label: 'Demo' },
  { id: 'gallery',   label: 'Gallery' },
  { id: 'embedding', label: 'Embedding Space' },
  { id: 'results',   label: 'Results' },
  { id: 'about',     label: 'About' },
];

export default function App() {
  const [tab, setTab]               = useState('demo');
  const [telemetry, setTelemetry]   = useState(null);
  const [compare, setCompare]       = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(setTelemetry)
      .catch(() => {});
  }, []);

  // Update navbar live stats after every query
  const handleQuery = result => {
    if (result?.latency_telemetry) {
      setTelemetry(prev => ({
        ...prev,
        total_latency_ms:     result.latency_telemetry.total_latency_ms,
        gallery_size:         prev?.gallery_size,
        vram_allocated_mb:    prev?.vram_allocated_mb,
      }));
    }
  };

  return (
    <div className="app">
      <Navbar
        tabs={TABS}
        active={tab}
        setTab={setTab}
        telemetry={telemetry}
        onMenuToggle={() => setSidebarOpen(v => !v)}
        sidebarOpen={sidebarOpen}
      />
      <main className="page">
        {tab === 'demo'      && (
          <RetrievalWorkspace
            onQuery={handleQuery}
            onCompare={setCompare}
            sidebarOpen={sidebarOpen}
          />
        )}
        {tab === 'gallery'   && <GalleryBrowser />}
        {tab === 'embedding' && <EmbeddingVisualizer />}
        {tab === 'results'   && <ResultsDashboard />}
        {tab === 'about'     && <AboutPanel />}
      </main>
      {compare && (
        <CompareModal
          query={compare.query}
          candidate={compare.candidate}
          onClose={() => setCompare(null)}
        />
      )}
    </div>
  );
}
