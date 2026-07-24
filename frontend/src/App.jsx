import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import OverviewSection from './components/OverviewSection';
import ArchitectureFlow from './components/ArchitectureFlow';
import RetrievalWorkspace from './components/RetrievalWorkspace';
import ExplainabilityLab from './components/ExplainabilityLab';
import BridgeAblationStudio from './components/BridgeAblationStudio';
import BenchmarkDashboard from './components/BenchmarkDashboard';
import DatasetExplorer from './components/DatasetExplorer';
import SystemTelemetry from './components/SystemTelemetry';
import EmbeddingSpaceInspector from './components/EmbeddingSpaceInspector';
import ReportExporter from './components/ReportExporter';
import CompareModal from './components/CompareModal';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [telemetry, setTelemetry] = useState(null);
  const [compareData, setCompareData] = useState(null);

  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => setTelemetry(data))
      .catch((err) => console.error("Error fetching health telemetry:", err));
  }, []);

  const handleQueryExecuted = (retrievalResult) => {
    if (retrievalResult && retrievalResult.latency_telemetry) {
      setTelemetry((prev) => ({
        ...prev,
        total_latency_ms: retrievalResult.latency_telemetry.total_latency_ms
      }));
    }
  };

  const handleCompareSelect = (query, candidate) => {
    setCompareData({ query, candidate });
  };

  return (
    <div className="app-container">
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        telemetry={telemetry} 
      />

      <main className="main-content-layout">
        {activeTab === 'overview' && (
          <OverviewSection onStartDemo={() => setActiveTab('workspace')} />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureFlow />
        )}

        {activeTab === 'workspace' && (
          <RetrievalWorkspace 
            onQueryExecuted={handleQueryExecuted} 
            onCompareSelect={handleCompareSelect} 
          />
        )}

        {activeTab === 'explainability' && (
          <ExplainabilityLab />
        )}

        {activeTab === 'ablation' && (
          <BridgeAblationStudio />
        )}

        {activeTab === 'benchmark' && (
          <BenchmarkDashboard />
        )}

        {activeTab === 'dataset' && (
          <DatasetExplorer />
        )}

        {activeTab === 'telemetry' && (
          <SystemTelemetry />
        )}

        {activeTab === 'embedding' && (
          <EmbeddingSpaceInspector />
        )}

        {activeTab === 'report' && (
          <ReportExporter />
        )}
      </main>

      {compareData && (
        <CompareModal 
          query={compareData.query} 
          candidate={compareData.candidate} 
          onClose={() => setCompareData(null)} 
        />
      )}
    </div>
  );
}
