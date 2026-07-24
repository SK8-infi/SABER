import React, { useState, useEffect } from 'react';
import { GitCompare, ArrowRight, ShieldCheck, AlertCircle, RefreshCw } from 'lucide-react';

export default function BridgeAblationStudio() {
  const [ablationData, setAblationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [queryIndex, setQueryIndex] = useState(0);

  const runAblation = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/retrieval/ablation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: 'ben14k',
          query_index: queryIndex,
          source_modality: 's1',
          target_modality: 's2',
          top_k: 5
        })
      });
      const data = await res.json();
      setAblationData(data);
    } catch (err) {
      console.error("Ablation query failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runAblation();
  }, [queryIndex]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      
      {/* Header Banner */}
      <div className="scientific-card">
        <div className="card-header">
          <span className="card-title">
            <GitCompare className="card-title-icon" size={16} /> Stochastic Latent Bridge Ablation Studio (Bridge OFF vs Bridge ON)
          </span>
          <span className="mono-tag saffron">ISRO BAH 2026 ABLATION DEMO</span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          This studio directly demonstrates the scientific contribution of SABER's <strong>Conditional Flow Matching (CFM) ODE Latent Bridge</strong>. 
          By executing retrieval simultaneously with <strong>Bridge OFF</strong> (baseline projection) vs <strong>Bridge ON</strong> (SABER ODE flow matching), 
          judges can observe candidate rank shifts and metric performance jumps.
        </p>
      </div>

      {/* Query Selector & Global Quantitative Jump Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px' }}>
        
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title" style={{ fontSize: '0.85rem' }}>Query Selection</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>QUERY SCENE INDEX:</label>
              <input 
                type="number" 
                className="scientific-input" 
                value={queryIndex} 
                min="0" 
                max="2965" 
                onChange={(e) => setQueryIndex(parseInt(e.target.value) || 0)} 
              />
            </div>
            <button className="secondary-btn" onClick={() => setQueryIndex(Math.floor(Math.random() * 2000))} style={{ justifyContent: 'center' }}>
              <RefreshCw size={12} /> RANDOM SAR QUERY
            </button>
          </div>
        </div>

        {ablationData && (
          <div className="scientific-card" style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', backgroundColor: 'var(--bg-card-hover)' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>CROSS-MODAL F1@5</div>
              <div className="data-mono" style={{ fontSize: '1.1rem', color: 'var(--text-secondary)' }}>
                {ablationData.delta.f1_at_5_baseline} <ArrowRight size={12} /> <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{ablationData.delta.f1_at_5_saber}</span>
              </div>
            </div>

            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>GLOBAL mAP BOOST</div>
              <div className="data-mono" style={{ fontSize: '1.1rem', color: 'var(--text-secondary)' }}>
                {ablationData.delta.map_baseline} <ArrowRight size={12} /> <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{ablationData.delta.map_saber}</span>
              </div>
            </div>

            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>QUERY AVG JACCARD DELTA</div>
              <div className="data-mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-saffron)' }}>
                +{ablationData.delta.jaccard_improvement}%
              </div>
            </div>
          </div>
        )}

      </div>

      {/* Side-by-Side Dual Retrieval Grid */}
      {ablationData && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          
          {/* BRANCH 1: BRIDGE OFF */}
          <div className="scientific-card" style={{ borderTop: '4px solid var(--accent-red)' }}>
            <div className="card-header">
              <span className="card-title" style={{ color: 'var(--accent-red)' }}>
                <AlertCircle size={16} /> Baseline Branch (Bridge OFF)
              </span>
              <span className="mono-tag">UNBRIDGED DOMAIN SHIFT</span>
            </div>

            <div style={{ backgroundColor: 'var(--bg-dark)', padding: '10px', borderRadius: '6px', marginBottom: '14px', fontSize: '0.8rem' }}>
              Avg Candidate Similarity: <span className="data-mono" style={{ color: 'var(--text-primary)' }}>{ablationData.bridge_off.avg_similarity}%</span> | 
              Avg Jaccard: <span className="data-mono" style={{ color: 'var(--text-primary)' }}>{ablationData.bridge_off.avg_jaccard}%</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {ablationData.bridge_off.candidates.map((cand) => (
                <div key={cand.rank} style={{ display: 'flex', alignItems: 'center', gap: '12px', backgroundColor: 'var(--bg-dark)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                  <span className="data-mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', width: '60px' }}>RANK #{cand.rank}</span>
                  <div style={{ flex: 1, fontSize: '0.78rem', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{cand.name}</div>
                  <span className="data-mono" style={{ fontSize: '0.78rem', color: 'var(--accent-saffron)' }}>{cand.similarity_score}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* BRANCH 2: BRIDGE ON */}
          <div className="scientific-card" style={{ borderTop: '4px solid var(--accent-green)' }}>
            <div className="card-header">
              <span className="card-title" style={{ color: 'var(--accent-green)' }}>
                <ShieldCheck size={16} /> SABER Branch (Bridge ON + CFM ODE)
              </span>
              <span className="mono-tag green">FLOW MATCHING ALIGNED</span>
            </div>

            <div style={{ backgroundColor: 'var(--bg-dark)', padding: '10px', borderRadius: '6px', marginBottom: '14px', fontSize: '0.8rem' }}>
              Avg Candidate Similarity: <span className="data-mono" style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{ablationData.bridge_on.avg_similarity}%</span> | 
              Avg Jaccard: <span className="data-mono" style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{ablationData.bridge_on.avg_jaccard}%</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {ablationData.bridge_on.candidates.map((cand) => (
                <div key={cand.rank} style={{ display: 'flex', alignItems: 'center', gap: '12px', backgroundColor: 'var(--bg-dark)', padding: '8px 12px', borderRadius: '6px', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                  <span className="data-mono" style={{ fontSize: '0.8rem', color: 'var(--accent-green)', fontWeight: 700, width: '60px' }}>RANK #{cand.rank}</span>
                  <div style={{ flex: 1, fontSize: '0.78rem', color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{cand.name}</div>
                  <span className="data-mono" style={{ fontSize: '0.78rem', color: 'var(--accent-green)', fontWeight: 700 }}>{cand.similarity_score}%</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
