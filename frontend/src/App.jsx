import { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import RetrievalWorkspace from './components/RetrievalWorkspace';
import ResultsDashboard from './components/ResultsDashboard';
import AboutPanel from './components/AboutPanel';
import CompareModal from './components/CompareModal';

const TABS = [
  { id: 'demo',    label: 'Demo' },
  { id: 'results', label: 'Results' },
  { id: 'about',   label: 'About' },
];

export default function App() {
  const [tab, setTab]         = useState('demo');
  const [telemetry, setTelemetry] = useState(null);
  const [compare, setCompare] = useState(null);

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(setTelemetry)
      .catch(() => {});
  }, []);

  const handleQuery = result => {
    if (result?.latency_telemetry?.total_latency_ms) {
      setTelemetry(prev => ({ ...prev, total_latency_ms: result.latency_telemetry.total_latency_ms }));
    }
  };

  return (
    <div className="app">
      <Navbar tabs={TABS} active={tab} setTab={setTab} telemetry={telemetry} />
      <main className="page">
        {tab === 'demo'    && <RetrievalWorkspace onQuery={handleQuery} onCompare={setCompare} />}
        {tab === 'results' && <ResultsDashboard />}
        {tab === 'about'   && <AboutPanel />}
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
