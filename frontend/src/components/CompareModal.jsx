import { X } from 'lucide-react';

export default function CompareModal({ query, candidate, onClose }) {
  if (!query || !candidate) return null;

  const items = [
    {
      img:     query.thumbnail,
      label:   query.source_modality.toUpperCase(),
      name:    query.name,
      classes: query.active_classes,
      tag:     'tag--saffron',
      border:  'rgba(255,153,51,0.25)',
      caption: 'QUERY IMAGE',
    },
    {
      img:     candidate.thumbnail,
      label:   `RANK #${candidate.rank}`,
      name:    candidate.name,
      classes: candidate.active_classes,
      tag:     'tag--green',
      border:  'rgba(34,197,94,0.25)',
      caption: 'RETRIEVED MATCH',
    },
  ];

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        {/* header */}
        <div className="modal-header">
          <div>
            <div className="card-title" style={{ fontSize: '0.9rem' }}>Multi-Sensor Inspector</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-2)', marginTop: 3 }}>
              Cross-modal pair comparison
            </div>
          </div>
          <button className="btn btn--ghost btn--sm" onClick={onClose}><X size={14} /></button>
        </div>

        {/* images */}
        <div className="modal-images">
          {items.map((item, i) => (
            <div key={i} className="modal-img-card" style={{ borderColor: item.border }}>
              <div className="modal-img-head">
                <span className={`tag ${item.tag}`}>{item.label}</span>
                <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-1)' }}>{item.caption}</span>
              </div>
              <img src={item.img} alt={item.name} className="modal-img" />
              <div style={{ padding: '10px 12px' }}>
                <div style={{ fontSize: '0.76rem', fontWeight: 600, marginBottom: 5 }}>{item.name}</div>
                <div className="chips">
                  {item.classes.map((cl, j) => <span className="chip" key={j}>{cl}</span>)}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* metrics */}
        <div className="modal-metrics">
          <div className="modal-metric">
            <div className="metric-label">Jaccard Overlap</div>
            <div className="mono text-cyan" style={{ fontSize: '1.4rem', fontWeight: 700 }}>
              {candidate.jaccard_overlap}%
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-2)', marginTop: 2 }}>
              semantic class similarity
            </div>
          </div>
          <div className="modal-metric-divider" />
          <div className="modal-metric">
            <div className="metric-label">Cosine Similarity</div>
            <div className="mono text-saffron" style={{ fontSize: '1.4rem', fontWeight: 700 }}>
              {candidate.similarity_score}%
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-2)', marginTop: 2 }}>
              embedding space distance
            </div>
          </div>
          <div className="modal-metric-divider" />
          <div className="modal-metric">
            <div className="metric-label">Rank Position</div>
            <div className="mono text-green" style={{ fontSize: '1.4rem', fontWeight: 700 }}>
              #{candidate.rank}
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-2)', marginTop: 2 }}>
              in gallery of 11,866
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
