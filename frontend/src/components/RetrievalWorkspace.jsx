import React, { useState, useEffect } from 'react';
import { Search, Sliders, Play, Layers, ChevronDown, ChevronUp, Clock, CheckCircle, RefreshCw, Eye } from 'lucide-react';

export default function RetrievalWorkspace({ onQueryExecuted, onCompareSelect }) {
  const [datasetName, setDatasetName] = useState('ben14k');
  const [sourceModality, setSourceModality] = useState('s1');
  const [targetModality, setTargetModality] = useState('s2');
  const [queryIndex, setQueryIndex] = useState(0);
  const [topK, setTopK] = useState(5);
  
  // Advanced Settings Collapsible State
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [enableBridge, setEnableBridge] = useState(true);
  const [enableRerank, setEnableRerank] = useState(false);
  const [odeSteps, setOdeSteps] = useState(5);
  
  // Query Results State
  const [loading, setLoading] = useState(false);
  const [retrievalData, setRetrievalData] = useState(null);

  const runQuery = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/retrieval/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: datasetName,
          query_index: queryIndex,
          source_modality: sourceModality,
          target_modality: targetModality,
          top_k: topK,
          enable_bridge: enableBridge,
          enable_rerank: enableRerank,
          ode_steps: odeSteps
        })
      });
      const data = await res.json();
      setRetrievalData(data);
      if (onQueryExecuted) onQueryExecuted(data);
    } catch (err) {
      console.error("Retrieval query failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runQuery();
  }, [queryIndex, sourceModality, targetModality, topK]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '20px', width: '100%' }}>
      
      {/* LEFT: QUERY CONTROL PANEL */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        
        {/* Modality & Pair Selector */}
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title">
              <Search className="card-title-icon" size={16} /> Retrieval Task & Pair
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                DATASET SELECTION
              </label>
              <select 
                className="scientific-select" 
                value={datasetName} 
                onChange={(e) => {
                  setDatasetName(e.target.value);
                  if (e.target.value === 'dsrsid') {
                    setSourceModality('pan');
                    setTargetModality('ms');
                  } else {
                    setSourceModality('s1');
                    setTargetModality('s2');
                  }
                }}
              >
                <option value="ben14k">BEN-14K (Sentinel-1 SAR / Sentinel-2 MS)</option>
                <option value="dsrsid">DSRSID (Gaofen-1 PAN / MS)</option>
              </select>
            </div>

            {datasetName === 'ben14k' ? (
              <>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                    SOURCE MODALITY (QUERY)
                  </label>
                  <select className="scientific-select" value={sourceModality} onChange={(e) => setSourceModality(e.target.value)}>
                    <option value="s1">Sentinel-1 SAR (2 Channels, C-Band)</option>
                    <option value="s2">Sentinel-2 Optical (12 Channels, VNIR/SWIR)</option>
                  </select>
                </div>

                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                    TARGET GALLERY MODALITY
                  </label>
                  <select className="scientific-select" value={targetModality} onChange={(e) => setTargetModality(e.target.value)}>
                    <option value="s2">Sentinel-2 Optical (Cross-Modal Search)</option>
                    <option value="s1">Sentinel-1 SAR (Same-Modal or Reverse Search)</option>
                  </select>
                </div>
              </>
            ) : (
              <>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                    SOURCE MODALITY (QUERY)
                  </label>
                  <select className="scientific-select" value={sourceModality} onChange={(e) => setSourceModality(e.target.value)}>
                    <option value="pan">Gaofen-1 Panchromatic (1 Channel, 2.5m)</option>
                    <option value="ms">Gaofen-1 Multispectral (4 Channels, 8m)</option>
                  </select>
                </div>

                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                    TARGET GALLERY MODALITY
                  </label>
                  <select className="scientific-select" value={targetModality} onChange={(e) => setTargetModality(e.target.value)}>
                    <option value="ms">Gaofen-1 Multispectral (Cross-Modal Search)</option>
                    <option value="pan">Gaofen-1 Panchromatic (Same-Modal Search)</option>
                  </select>
                </div>
              </>
            )}

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                QUERY SCENE INDEX: <span className="data-mono" style={{ color: 'var(--accent-saffron)' }}>#{queryIndex}</span>
              </label>
              <input 
                type="number" 
                className="scientific-input" 
                min="0" 
                max="2965" 
                value={queryIndex} 
                onChange={(e) => setQueryIndex(parseInt(e.target.value) || 0)} 
              />
              <button 
                className="secondary-btn" 
                style={{ width: '100%', marginTop: '6px', justifyContent: 'center' }}
                onClick={() => setQueryIndex(Math.floor(Math.random() * 2000))}
              >
                <RefreshCw size={12} /> RANDOM SAMPLE
              </button>
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                TOP-K CANDIDATES: <span className="data-mono" style={{ color: 'var(--accent-cyan)' }}>{topK}</span>
              </label>
              <input 
                type="range" 
                min="1" 
                max="20" 
                value={topK} 
                onChange={(e) => setTopK(parseInt(e.target.value))} 
                style={{ width: '100%', accentColor: 'var(--accent-saffron)' }}
              />
            </div>
          </div>
        </div>

        {/* Collapsible Advanced Research Settings */}
        <div className="scientific-card">
          <div 
            className="card-header" 
            style={{ cursor: 'pointer', marginBottom: showAdvanced ? '16px' : '0' }}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <span className="card-title" style={{ fontSize: '0.85rem' }}>
              <Sliders size={14} /> Advanced Research Parameters
            </span>
            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </div>

          {showAdvanced && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', paddingTop: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>CFM Latent Bridge</span>
                <input 
                  type="checkbox" 
                  checked={enableBridge} 
                  onChange={(e) => setEnableBridge(e.target.checked)} 
                  style={{ accentColor: 'var(--accent-saffron)' }}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                  ODE EULER SOLVER STEPS: {odeSteps}
                </label>
                <input 
                  type="range" 
                  min="1" 
                  max="15" 
                  value={odeSteps} 
                  onChange={(e) => setOdeSteps(parseInt(e.target.value))} 
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Soft Jaccard Reranking</span>
                <input 
                  type="checkbox" 
                  checked={enableRerank} 
                  onChange={(e) => setEnableRerank(e.target.checked)} 
                  style={{ accentColor: 'var(--accent-saffron)' }}
                />
              </div>
            </div>
          )}
        </div>

        <button className="primary-action-btn" onClick={runQuery} disabled={loading}>
          {loading ? 'EXECUTING SEARCH...' : 'EXECUTE RETRIEVAL SEARCH'}
        </button>

      </div>

      {/* RIGHT: LIVE RETRIEVAL RESULTS WORKSPACE */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        
        {retrievalData && (
          <>
            {/* Top Query Scene Summary & Latency Metric Strip */}
            <div className="scientific-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ width: '64px', height: '64px', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-glow)' }}>
                  <img src={retrievalData.query.thumbnail} alt="Query" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </div>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="mono-tag saffron">{retrievalData.query.source_modality.toUpperCase()} QUERY</span>
                    <span style={{ fontSize: '0.9rem', fontWeight: 700 }}>{retrievalData.query.name}</span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    Target Classes: {retrievalData.query.active_classes.join(', ') || 'Unclassified'}
                  </div>
                </div>
              </div>

              {/* Latency Telemetry Highlights */}
              <div style={{ display: 'flex', gap: '20px', backgroundColor: 'var(--bg-dark)', padding: '10px 16px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>FEATURE EXT</div>
                  <div className="data-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>
                    {retrievalData.latency_telemetry.feature_extraction_ms} ms
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>CFM BRIDGE ODE</div>
                  <div className="data-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-saffron)' }}>
                    {retrievalData.latency_telemetry.latent_bridge_ms} ms
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>FAISS SEARCH</div>
                  <div className="data-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-blue)' }}>
                    {retrievalData.latency_telemetry.faiss_search_ms} ms
                  </div>
                </div>

                <div style={{ borderLeft: '1px solid var(--border-color)', paddingLeft: '12px' }}>
                  <div style={{ fontSize: '0.65rem', color: 'var(--accent-green)', fontWeight: 700 }}>END-TO-END TOTAL</div>
                  <div className="data-mono" style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                    {retrievalData.latency_telemetry.total_latency_ms} ms
                  </div>
                </div>
              </div>

            </div>

            {/* Candidates Grid */}
            <div className="scientific-card">
              <div className="card-header">
                <span className="card-title">
                  <Layers className="card-title-icon" size={16} /> Top-{topK} Ranked Retrieval Candidates ({retrievalData.query.target_modality.toUpperCase()} Gallery)
                </span>
                <span className="mono-tag green">MATCHES RETURNED: {retrievalData.candidates.length}</span>
              </div>

              <div className="candidate-grid">
                {retrievalData.candidates.map((cand) => (
                  <div className="candidate-card" key={cand.rank}>
                    <div className="rank-badge">RANK #{cand.rank}</div>
                    <div className="score-badge">{cand.similarity_score}% SIM</div>
                    
                    <div className="thumb-wrapper">
                      <img src={cand.thumbnail} alt={cand.name} className="thumb-img" />
                    </div>

                    <div className="candidate-meta">
                      <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {cand.name}
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        <span>Jaccard Overlap:</span>
                        <span className="data-mono" style={{ color: 'var(--accent-cyan)' }}>{cand.jaccard_overlap}%</span>
                      </div>

                      <div className="class-chips">
                        {cand.active_classes.slice(0, 3).map((cls, idx) => (
                          <span className="class-chip" key={idx}>{cls}</span>
                        ))}
                      </div>

                      <button 
                        className="secondary-btn" 
                        style={{ marginTop: '8px', padding: '4px 8px', fontSize: '0.7rem', justifyContent: 'center' }}
                        onClick={() => onCompareSelect(retrievalData.query, cand)}
                      >
                        <Eye size={12} /> COMPARE SPECTRUM
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

      </div>
    </div>
  );
}
