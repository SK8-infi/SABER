import React from 'react';
import { X, Layers, CheckCircle, AlertTriangle } from 'lucide-react';

export default function CompareModal({ query, candidate, onClose }) {
  if (!query || !candidate) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(11, 14, 20, 0.85)',
      backdropFilter: 'blur(8px)',
      zIndex: 200,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '24px'
    }}>
      <div className="scientific-card" style={{ width: '100%', maxWidth: '900px', maxHeight: '90vh', overflowY: 'auto' }}>
        
        <div className="card-header">
          <span className="card-title">
            <Layers className="card-title-icon" size={16} /> Side-by-Side Multi-Sensor Image & Spectrum Inspector
          </span>
          <button className="secondary-btn" onClick={onClose} style={{ padding: '4px 8px' }}>
            <X size={16} />
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
          
          {/* Left: Query Image */}
          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span className="mono-tag saffron">QUERY: {query.source_modality.toUpperCase()}</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{query.name}</span>
            </div>
            <div style={{ width: '100%', height: '240px', borderRadius: '6px', overflow: 'hidden', marginBottom: '12px' }}>
              <img src={query.thumbnail} alt="Query" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Active Classes: {query.active_classes.join(', ') || 'None'}
            </div>
          </div>

          {/* Right: Candidate Image */}
          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '16px', borderRadius: '6px', border: '1px solid var(--accent-saffron)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span className="mono-tag green">MATCH RANK #{candidate.rank} ({candidate.similarity_score}%)</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{candidate.name}</span>
            </div>
            <div style={{ width: '100%', height: '240px', borderRadius: '6px', overflow: 'hidden', marginBottom: '12px' }}>
              <img src={candidate.thumbnail} alt="Candidate" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Active Classes: {candidate.active_classes.join(', ') || 'None'}
            </div>
          </div>

        </div>

        {/* Overlap Summary */}
        <div style={{ backgroundColor: 'var(--bg-dark)', padding: '14px', borderRadius: '6px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>MULTI-LABEL JACCARD OVERLAP</div>
            <div className="data-mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
              {candidate.jaccard_overlap}% Overlap
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>COSINE VECTOR SIMILARITY</div>
            <div className="data-mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-saffron)' }}>
              {candidate.similarity_score}% Score
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
