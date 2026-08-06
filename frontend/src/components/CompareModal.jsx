import { X, Check, Minus } from 'lucide-react';

export default function CompareModal({ query, candidate, onClose }) {
  if (!query || !candidate) return null;

  const queryClasses     = query.active_classes     ?? [];
  const candidateClasses = candidate.active_classes ?? [];
  const sharedClasses    = queryClasses.filter(c => candidateClasses.includes(c));
  const onlyInQuery      = queryClasses.filter(c => !candidateClasses.includes(c));
  const onlyInResult     = candidateClasses.filter(c => !queryClasses.includes(c));

  const items = [
    {
      img:     query.thumbnail,
      label:   query.source_modality.toUpperCase(),
      name:    query.name,
      classes: queryClasses,
      tag:     'tag--saffron',
      border:  'rgba(255,153,51,0.25)',
      caption: 'QUERY IMAGE',
    },
    {
      img:     candidate.thumbnail,
      label:   `RANK #${candidate.rank}`,
      name:    candidate.name,
      classes: candidateClasses,
      tag:     'tag--green',
      border:  'rgba(34,197,94,0.25)',
      caption: 'RETRIEVED MATCH',
    },
  ];

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        {/* Header */}
        <div className="modal-header">
          <div>
            <div className="card-title" style={{ fontSize: '0.9rem' }}>Multi-Sensor Inspector</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-2)', marginTop: 3 }}>
              Cross-modal pair comparison
            </div>
          </div>
          <button className="btn btn--ghost btn--sm" onClick={onClose}><X size={14} /></button>
        </div>

        {/* Images */}
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
                  {item.classes.map((cl, j) => {
                    const isShared = sharedClasses.includes(cl);
                    return (
                      <span className={`chip${isShared ? ' chip--matched' : ''}`} key={j}>
                        {isShared && <Check size={8} strokeWidth={3} style={{ marginRight: 2 }} />}
                        {cl}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Metrics */}
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

        {/* Class breakdown — Venn-style */}
        <div className="modal-class-breakdown">
          <div className="class-breakdown-col class-breakdown-col--shared">
            <div className="class-breakdown-title">
              <Check size={11} color="var(--green)" /> Shared
              <span className="class-breakdown-count">{sharedClasses.length}</span>
            </div>
            <div className="chips" style={{ marginTop: 6 }}>
              {sharedClasses.length > 0
                ? sharedClasses.map((cl, i) => (
                    <span className="chip chip--matched" key={i}>{cl}</span>
                  ))
                : <span className="text-dim" style={{ fontSize: '0.7rem' }}>None</span>}
            </div>
          </div>
          <div className="class-breakdown-col class-breakdown-col--query">
            <div className="class-breakdown-title">
              <Minus size={11} color="var(--saffron)" /> Only in Query
              <span className="class-breakdown-count">{onlyInQuery.length}</span>
            </div>
            <div className="chips" style={{ marginTop: 6 }}>
              {onlyInQuery.length > 0
                ? onlyInQuery.map((cl, i) => (
                    <span className="chip chip--query" key={i}>{cl}</span>
                  ))
                : <span className="text-dim" style={{ fontSize: '0.7rem' }}>None</span>}
            </div>
          </div>
          <div className="class-breakdown-col class-breakdown-col--result">
            <div className="class-breakdown-title">
              <Minus size={11} color="var(--cyan)" /> Only in Result
              <span className="class-breakdown-count">{onlyInResult.length}</span>
            </div>
            <div className="chips" style={{ marginTop: 6 }}>
              {onlyInResult.length > 0
                ? onlyInResult.map((cl, i) => (
                    <span className="chip" style={{ background: 'var(--cyan-dim)', color: 'var(--cyan)', border: '1px solid rgba(0,229,255,0.2)' }} key={i}>{cl}</span>
                  ))
                : <span className="text-dim" style={{ fontSize: '0.7rem' }}>None</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
