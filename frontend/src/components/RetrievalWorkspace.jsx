import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Search, RefreshCw, ChevronDown, ChevronUp, Eye, Check, X,
  GitCompare, Zap, AlertCircle, ShieldCheck, ArrowRight,
  RotateCcw, Clock, History, Info,
} from 'lucide-react';

/* ─── ODE latency estimate (linear model from benchmark data) ── */
function odeLatencyEstimate(steps) {
  // ~14ms at 1 step, ~28ms at 5 steps — linear interpolation
  return (12 + steps * 3.3).toFixed(1);
}

/* ─── Skeleton card ─────────────────────────────────────────── */
function SkeletonCard() {
  return (
    <div className="cand-card skeleton-card">
      <div className="skel skel-thumb" />
      <div className="cand-body">
        <div className="skel skel-line" style={{ width: '80%' }} />
        <div className="skel skel-line" style={{ width: '100%', height: 4 }} />
        <div className="skel skel-line" style={{ width: '60%' }} />
        <div className="skel skel-line" style={{ width: '40%' }} />
      </div>
    </div>
  );
}

function SkeletonGrid({ count = 5 }) {
  return (
    <div className="cand-grid">
      {Array.from({ length: count }).map((_, i) => <SkeletonCard key={i} />)}
    </div>
  );
}

/* ─── Latency strip ─────────────────────────────────────────── */
function LatencyStrip({ lats }) {
  const items = [
    { label: 'Prep',     val: lats?.preprocessing_ms,      color: 'var(--text-2)' },
    { label: 'Feat Ext', val: lats?.feature_extraction_ms, color: 'var(--cyan)' },
    { label: 'CFM ODE',  val: lats?.latent_bridge_ms,      color: 'var(--saffron)', hideIfZero: true },
    { label: 'FAISS',    val: lats?.faiss_search_ms,       color: 'var(--blue)' },
    { label: 'Re-Rank',  val: lats?.rerank_ms,             color: '#a855f7', hideIfZero: true },
    { label: 'Total',    val: lats?.total_latency_ms,      color: 'var(--green)', bold: true },
  ].filter(m => !m.hideIfZero || (m.val && m.val > 0));

  return (
    <div className="lat-strip" style={{ gap: '0.4rem', flexWrap: 'wrap' }}>
      {items.map(m => (
        <div key={m.label} className={`lat-item${m.bold ? ' lat-item--total' : ''}`}>
          <span className="metric-label">{m.label}</span>
          <span className="mono" style={{ fontSize: '0.82rem', fontWeight: m.bold ? 700 : 500, color: m.color }}>
            {m.val !== undefined && m.val !== null ? `${m.val} ms` : 'N/A'}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ─── Similarity bar ────────────────────────────────────────── */
function SimilarityBar({ pct, color = 'var(--saffron)' }) {
  return (
    <div className="sim-bar-track">
      <div className="sim-bar-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

/* ─── Candidate card ────────────────────────────────────────── */
function CandidateCard({ c, query, onCompare, rank }) {
  const hasOverlap = c.jaccard_overlap > 0 || (
    query?.active_classes && c.active_classes?.some(cl => query.active_classes.includes(cl))
  );
  const hue = c.similarity_score >= 80 ? 'var(--green)'
    : c.similarity_score >= 60 ? 'var(--saffron)'
    : 'var(--red)';

  return (
    <div className={`cand-card${hasOverlap ? ' cand-card--match' : ' cand-card--no-match'}`}>
      <span className="cand-rank">#{rank}</span>
      <div
        className={`cand-match-badge ${hasOverlap ? 'cand-match-badge--yes' : 'cand-match-badge--no'}`}
        title={hasOverlap ? 'Ground-Truth Label Overlap' : 'No Label Overlap'}
      >
        {hasOverlap ? <Check size={11} strokeWidth={3} /> : <X size={11} strokeWidth={3} />}
      </div>
      <span className="cand-score" style={{ background: hue === 'var(--green)' ? 'var(--green)' : hue === 'var(--saffron)' ? 'var(--saffron)' : 'var(--red)' }}>
        {c.similarity_score}%
      </span>
      <img src={c.thumbnail} alt={c.name} className="cand-thumb" loading="lazy" />
      <div className="cand-body">
        <div className="cand-name">{c.name}</div>
        <SimilarityBar pct={c.similarity_score} color={hue} />
        <div className="list-row" style={{ marginTop: 2 }}>
          <span className="text-dim" style={{ fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: 4 }}>
            Jaccard
            {hasOverlap
              ? <span className="cand-status-tag cand-status-tag--yes"><Check size={8} strokeWidth={3} /> Match</span>
              : <span className="cand-status-tag cand-status-tag--no"><X size={8} strokeWidth={3} /> No Match</span>}
          </span>
          <span className="mono text-cyan" style={{ fontSize: '0.7rem' }}>{c.jaccard_overlap}%</span>
        </div>
        <div className="chips" style={{ marginTop: 2 }}>
          {c.active_classes.slice(0, 3).map((cl, i) => {
            const isShared = query?.active_classes?.includes(cl);
            return (
              <span className={`chip${isShared ? ' chip--matched' : ''}`} key={i}>
                {isShared && <Check size={8} strokeWidth={3} style={{ marginRight: 2 }} />}
                {cl}
              </span>
            );
          })}
        </div>
        <button
          className="btn btn--ghost btn--sm"
          style={{ marginTop: 6, width: '100%' }}
          onClick={() => onCompare({ query, candidate: c })}
        >
          <Eye size={11} /> Inspect
        </button>
      </div>
    </div>
  );
}

/* ─── Ablation candidate row (with thumbnail) ───────────────── */
function AblationRow({ c, color }) {
  return (
    <div className="abl-row">
      <span className="mono text-dim abl-rank">#{c.rank}</span>
      {c.thumbnail && (
        <img src={c.thumbnail} alt={c.name} className="abl-thumb" loading="lazy" />
      )}
      <span className="abl-row-name">{c.name}</span>
      <SimilarityBar pct={c.similarity_score} color={color} />
      <span className="mono" style={{ fontSize: '0.72rem', color, fontWeight: 700, flexShrink: 0 }}>
        {c.similarity_score}%
      </span>
    </div>
  );
}

/* ─── Query history panel ───────────────────────────────────── */
function HistoryPanel({ history, onReplay }) {
  if (!history.length) return null;
  return (
    <div className="card history-panel">
      <div className="card-head">
        <span className="card-title"><History size={13} /> Recent Queries</span>
        <span className="tag">{history.length} runs</span>
      </div>
      <div className="history-list">
        {history.map((h, i) => (
          <div key={i} className="history-row" onClick={() => onReplay(h)}>
            <span className="mono text-dim" style={{ fontSize: '0.65rem', flexShrink: 0 }}>#{h.qIdx}</span>
            <span className="history-mods">
              <span className="tag tag--saffron" style={{ fontSize: '0.58rem' }}>{h.srcMod.toUpperCase()}</span>
              <ArrowRight size={10} color="var(--text-3)" />
              <span className="tag tag--cyan" style={{ fontSize: '0.58rem' }}>{h.tgtMod.toUpperCase()}</span>
            </span>
            <span className="mono text-dim" style={{ fontSize: '0.65rem' }}>{h.dataset}</span>
            <span className="mono" style={{ fontSize: '0.65rem', color: 'var(--green)', marginLeft: 'auto' }}>
              {h.latency} ms
            </span>
            <span className="mono text-saffron" style={{ fontSize: '0.65rem' }}>
              top: {h.topScore}%
            </span>
            <button className="btn btn--ghost btn--sm history-replay" title="Replay this query">
              <RotateCcw size={10} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}


/* ─── QUERY MODE ────────────────────────────────────────────── */
function QueryMode({ params, onResult, onCompare, onHistoryAdd }) {
  const { dataset, srcMod, tgtMod, qIdx, topK, bridge, rerank, odeSteps } = params;
  const [loading, setLoading]         = useState(false);
  const [resultSaber, setResultSaber] = useState(null);
  const [resultIsro,  setResultIsro]  = useState(null);
  const [error, setError]             = useState(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [resSaber, resIsro] = await Promise.all([
        fetch('/api/retrieval/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_name:    dataset,
            query_index:     qIdx,
            source_modality: srcMod,
            target_modality: tgtMod,
            model_name:      'saber',
            top_k:           topK,
            enable_bridge:   bridge,
            enable_rerank:   rerank,
            ode_steps:       odeSteps,
          }),
        }),
        fetch('/api/retrieval/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_name:    dataset,
            query_index:     qIdx,
            source_modality: srcMod,
            target_modality: tgtMod,
            model_name:      'isro_official',
            top_k:           topK,
            enable_bridge:   false,
            enable_rerank:   false,
            ode_steps:       odeSteps,
          }),
        }),
      ]);
      if (!resSaber.ok) throw new Error(`SABER query failed: HTTP ${resSaber.status}`);
      if (!resIsro.ok)  throw new Error(`ISRO query failed: HTTP ${resIsro.status}`);
      const dataSaber = await resSaber.json();
      const dataIsro  = await resIsro.json();
      setResultSaber(dataSaber);
      setResultIsro(dataIsro);
      if (onResult) onResult(dataSaber);
      if (onHistoryAdd) onHistoryAdd({
        qIdx, srcMod, tgtMod, dataset,
        latency:  dataSaber.latency_telemetry?.total_latency_ms ?? '—',
        topScore: dataSaber.candidates?.[0]?.similarity_score ?? '—',
        timestamp: Date.now(),
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [dataset, qIdx, srcMod, tgtMod, topK, bridge, rerank, odeSteps]);

  useEffect(() => { run(); }, [dataset, qIdx, srcMod, tgtMod, topK]);

  const queryInfo = resultSaber?.query || resultIsro?.query;

  return (
    <div className="results-area">
      {/* Loading — skeleton */}
      {loading && (
        <>
          <div className="query-header-skeleton">
            <div className="skel skel-thumb-lg" />
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div className="skel skel-line" style={{ width: '50%' }} />
              <div className="skel skel-line" style={{ width: '75%' }} />
              <div className="skel skel-line" style={{ width: '40%' }} />
            </div>
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="spinner" style={{ width: 14, height: 14 }} />
            Running dual-model retrieval…
          </div>
          <SkeletonGrid count={topK} />
          <SkeletonGrid count={topK} />
        </>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="error-state" style={{ flexDirection: 'column', gap: 12, padding: '40px 24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertCircle size={16} color="var(--red)" />
            <span style={{ color: 'var(--red)', fontSize: '0.82rem' }}>{error}</span>
          </div>
          <button className="btn btn--ghost btn--sm" onClick={run}>
            <RotateCcw size={12} /> Retry
          </button>
        </div>
      )}

      {/* Results */}
      {queryInfo && !loading && (
        <>
          <div className="query-header">
            <div className="query-img-wrap">
              <img src={queryInfo.thumbnail} alt="Query" className="query-thumb" />
              <span className="query-mod-badge">{queryInfo.source_modality.toUpperCase()}</span>
            </div>
            <div className="query-meta">
              <div className="query-name">{queryInfo.name}</div>
              <div className="chips" style={{ marginTop: 4 }}>
                {queryInfo.active_classes.slice(0, 6).map((cl, i) => (
                  <span className="chip chip--query" key={i}>{cl}</span>
                ))}
              </div>
              <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-2)' }}>
                Target: <span className="text-dim">{queryInfo.target_modality.toUpperCase()}</span>
                &nbsp;·&nbsp;Top-{topK}
                &nbsp;·&nbsp;Bridge: <span style={{ color: bridge ? 'var(--green)' : 'var(--red)' }}>{bridge ? 'ON' : 'OFF'}</span>
              </div>
            </div>
          </div>

          {/* MODEL 1: SABER */}
          {resultSaber && (
            <div className="model-results-section" style={{ marginBottom: 28 }}>
              <div className="section-title-bar">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span className="model-badge model-badge--saber">MODEL 1</span>
                  <span style={{ fontWeight: 600, color: '#fff', fontSize: '0.95rem' }}>
                    SABER — Neural ODE Bridge (Ours)
                  </span>
                </div>
                <LatencyStrip lats={resultSaber.latency_telemetry} />
              </div>
              <div className="cand-grid">
                {resultSaber.candidates.map(c => (
                  <CandidateCard key={`saber_${c.rank}`} c={c} query={resultSaber.query} onCompare={onCompare} rank={c.rank} />
                ))}
              </div>
            </div>
          )}

          {/* MODEL 2: ISRO */}
          {resultIsro && (
            <div className="model-results-section">
              <div className="section-title-bar" style={{ borderBottomColor: 'rgba(160,100,255,0.2)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span className="model-badge model-badge--isro">MODEL 2</span>
                  <span style={{ fontWeight: 600, color: '#fff', fontSize: '0.95rem' }}>
                    ISRO Official Best <span style={{ fontSize: '0.75rem', opacity: 0.6, fontWeight: 400 }}>(best_ben14k_isro_retrieval.pt)</span>
                  </span>
                </div>
                <LatencyStrip lats={resultIsro.latency_telemetry} />
              </div>
              <div className="cand-grid">
                {resultIsro.candidates.map(c => (
                  <CandidateCard key={`isro_${c.rank}`} c={c} query={resultIsro.query} onCompare={onCompare} rank={c.rank} />
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {!resultSaber && !resultIsro && !loading && !error && (
        <div className="empty-state">
          <Search size={32} color="var(--border-2)" />
          <span className="text-dim">Configure params and hit Execute</span>
        </div>
      )}
    </div>
  );
}


/* ─── ABLATION MODE ─────────────────────────────────────────── */
function AblationMode({ params }) {
  const { dataset, qIdx } = params;
  const [loading, setLoading] = useState(false);
  const [data, setData]       = useState(null);
  const [error, setError]     = useState(null);

  const isBen = dataset === 'ben14k';
  const srcMod = isBen ? 's1' : 'pan';
  const tgtMod = isBen ? 's2' : 'ms';

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/retrieval/ablation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_name: dataset, query_index: qIdx, source_modality: srcMod, target_modality: tgtMod, top_k: 5 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [dataset, qIdx]);

  useEffect(() => { run(); }, [dataset, qIdx]);

  return (
    <div className="results-area">
      {loading && (
        <>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="spinner" style={{ width: 14, height: 14 }} />
            Running ablation study (Bridge OFF vs Bridge ON)…
          </div>
          <div className="ablation-cols">
            <SkeletonGrid count={5} />
            <SkeletonGrid count={5} />
          </div>
        </>
      )}

      {error && !loading && (
        <div className="error-state" style={{ flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertCircle size={16} color="var(--red)" />
            <span style={{ color: 'var(--red)', fontSize: '0.82rem' }}>{error}</span>
          </div>
          <button className="btn btn--ghost btn--sm" onClick={run}>
            <RotateCcw size={12} /> Retry
          </button>
        </div>
      )}

      {data && !loading && (
        <>
          {/* Delta summary */}
          <div className="ablation-delta">
            {[
              { label: 'F1@5', base: data.delta.f1_at_5_baseline, saber: data.delta.f1_at_5_saber },
              { label: 'mAP',  base: data.delta.map_baseline,     saber: data.delta.map_saber },
            ].map(m => (
              <div key={m.label} className="delta-item">
                <span className="metric-label">{m.label}</span>
                <div className="delta-values mono">
                  <span style={{ color: 'var(--red)' }}>{m.base}</span>
                  <ArrowRight size={11} color="var(--text-2)" />
                  <span style={{ color: 'var(--green)', fontWeight: 700 }}>{m.saber}</span>
                </div>
              </div>
            ))}
            <div className="delta-item">
              <span className="metric-label">Sim Δ</span>
              <span className="mono text-saffron" style={{ fontSize: '1rem', fontWeight: 700 }}>
                +{data.delta.similarity_improvement}%
              </span>
            </div>
            <div className="delta-item">
              <span className="metric-label">Jaccard Δ</span>
              <span className="mono text-green" style={{ fontSize: '1rem', fontWeight: 700 }}>
                +{data.delta.jaccard_improvement}%
              </span>
            </div>
          </div>

          {/* Side-by-side with thumbnails */}
          <div className="ablation-cols">
            <div className="ablation-col ablation-col--off">
              <div className="abl-col-head">
                <span className="abl-col-title text-red"><AlertCircle size={13} /> Baseline — Bridge OFF</span>
                <div className="abl-col-stats mono">
                  <span>Sim <span style={{ color: 'var(--text-0)' }}>{data.bridge_off.avg_similarity}%</span></span>
                  <span>Jac <span style={{ color: 'var(--text-0)' }}>{data.bridge_off.avg_jaccard}%</span></span>
                </div>
              </div>
              <div className="abl-list">
                {data.bridge_off.candidates.map(c => (
                  <AblationRow key={c.rank} c={c} color="var(--red)" />
                ))}
              </div>
            </div>

            <div className="ablation-col ablation-col--on">
              <div className="abl-col-head">
                <span className="abl-col-title text-green"><ShieldCheck size={13} /> SABER — Bridge ON</span>
                <div className="abl-col-stats mono">
                  <span>Sim <span style={{ color: 'var(--green)', fontWeight: 700 }}>{data.bridge_on.avg_similarity}%</span></span>
                  <span>Jac <span style={{ color: 'var(--green)', fontWeight: 700 }}>{data.bridge_on.avg_jaccard}%</span></span>
                </div>
              </div>
              <div className="abl-list">
                {data.bridge_on.candidates.map(c => (
                  <AblationRow key={c.rank} c={c} color="var(--green)" />
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}


/* ─── SCENE PREVIEW ─────────────────────────────────────────── */
function ScenePreview({ qIdx, dataset, srcMod }) {
  const [thumb, setThumb] = useState(null);

  useEffect(() => {
    setThumb(null);
    const page = Math.floor(qIdx / 12) + 1;
    const offset = qIdx % 12;
    fetch(`/api/dataset/samples?dataset_name=${dataset}&page=${page}&limit=12`)
      .then(r => r.json())
      .then(d => {
        const item = d.items?.[offset];
        if (item?.thumbnail) setThumb(item.thumbnail);
      })
      .catch(() => {});
  }, [qIdx, dataset]);

  if (!thumb) return (
    <div className="scene-preview-placeholder">
      <div className="skel" style={{ width: '100%', height: '100%', borderRadius: 4 }} />
    </div>
  );

  return (
    <div className="scene-preview-wrap">
      <img src={thumb} alt={`Scene #${qIdx}`} className="scene-preview-img" />
      <span className="scene-preview-badge">{srcMod.toUpperCase()}</span>
    </div>
  );
}

/* ─── MAIN COMPONENT ────────────────────────────────────────── */
export default function RetrievalWorkspace({ onQuery, onCompare, sidebarOpen }) {
  const [mode, setMode]       = useState('query');
  const [dataset, setDataset] = useState('ben14k');
  const [srcMod, setSrcMod]   = useState('s1');
  const [tgtMod, setTgtMod]   = useState('s2');
  const [qIdx, setQIdx]       = useState(0);
  const [topK, setTopK]       = useState(5);
  const [showAdv, setShowAdv] = useState(false);
  const [bridge, setBridge]   = useState(true);
  const [rerank, setRerank]   = useState(true);
  const [odeSteps, setOdeSteps] = useState(3);
  const [history, setHistory] = useState([]);
  const isBen = dataset === 'ben14k';

  const handleDatasetChange = v => {
    setDataset(v);
    if (v === 'dsrsid') { setSrcMod('pan'); setTgtMod('ms'); }
    else                { setSrcMod('s1');  setTgtMod('s2'); }
  };

  const randomize = () => setQIdx(Math.floor(Math.random() * 2000));

  const addToHistory = entry => {
    setHistory(prev => [entry, ...prev].slice(0, 8));
  };

  const replayQuery = h => {
    setDataset(h.dataset);
    setSrcMod(h.srcMod);
    setTgtMod(h.tgtMod);
    setQIdx(h.qIdx);
    setMode('query');
  };

  const params = { dataset, srcMod, tgtMod, qIdx, topK, bridge, rerank, odeSteps };

  return (
    <div className={`workspace${sidebarOpen ? '' : ' workspace--collapsed'}`}>

      {/* ── LEFT SIDEBAR ─────────────────────── */}
      <aside className={`sidebar${sidebarOpen ? '' : ' sidebar--hidden'}`}>

        {/* Mode toggle */}
        <div className="mode-toggle">
          <button className={`mode-btn${mode === 'query' ? ' mode-btn--active' : ''}`} onClick={() => setMode('query')}>
            <Search size={12} /> Query
          </button>
          <button className={`mode-btn${mode === 'ablation' ? ' mode-btn--active' : ''}`} onClick={() => setMode('ablation')}>
            <GitCompare size={12} /> Ablation
          </button>
        </div>

        {/* Dataset */}
        <div className="sidebar-section">
          <div className="field">
            <label className="field-label">Dataset</label>
            <select className="select" value={dataset} onChange={e => handleDatasetChange(e.target.value)}>
              <option value="ben14k">BEN-14K — Sentinel-1/2</option>
              <option value="dsrsid">DSRSID — Gaofen-1</option>
            </select>
          </div>
        </div>

        {/* Modality selectors */}
        {mode === 'query' && (
          <div className="sidebar-section">
            <div className="field">
              <label className="field-label">
                Source
                <span className="field-hint" title={isBen ? 'C-band SAR radar, 2ch VV+VH' : 'Panchromatic 2.5m, 1ch'}>
                  <Info size={10} />
                </span>
              </label>
              <select className="select" value={srcMod} onChange={e => setSrcMod(e.target.value)}>
                {isBen
                  ? <><option value="s1">Sentinel-1 SAR (2ch)</option><option value="s2">Sentinel-2 MS (12ch)</option></>
                  : <><option value="pan">Gaofen-1 PAN (1ch)</option><option value="ms">Gaofen-1 MS (4ch)</option></>}
              </select>
            </div>
            <div className="field" style={{ marginTop: 10 }}>
              <label className="field-label">
                Target Gallery
                <span className="field-hint" title={isBen ? 'VNIR+SWIR multispectral, 12ch' : 'RGB+NIR multispectral, 4ch'}>
                  <Info size={10} />
                </span>
              </label>
              <select className="select" value={tgtMod} onChange={e => setTgtMod(e.target.value)}>
                {isBen
                  ? <><option value="s2">Sentinel-2 MS (Cross)</option><option value="s1">Sentinel-1 SAR (Same)</option></>
                  : <><option value="ms">Gaofen-1 MS (Cross)</option><option value="pan">Gaofen-1 PAN (Same)</option></>}
              </select>
            </div>
          </div>
        )}

        {mode === 'ablation' && (
          <div className="sidebar-section">
            <div className="mode-info">
              <Zap size={11} color="var(--saffron)" />
              <span>Locked to cross-modal — SAR → Optical</span>
            </div>
          </div>
        )}

        {/* Scene index + preview */}
        <div className="sidebar-section">
          <div className="field">
            <label className="field-label">
              Scene Index&nbsp;
              <span className="mono text-saffron">#{qIdx}</span>
            </label>
            <div className="scene-index-row">
              <input
                type="number" className="input" min="0" max="2965"
                value={qIdx}
                onChange={e => setQIdx(parseInt(e.target.value) || 0)}
                style={{ flex: 1 }}
              />
              <ScenePreview qIdx={qIdx} dataset={dataset} srcMod={srcMod} />
            </div>
          </div>
          <button className="btn btn--ghost btn--sm btn--full" style={{ marginTop: 8 }} onClick={randomize}>
            <RefreshCw size={11} /> Random Scene
          </button>
        </div>

        {/* Top-K */}
        {mode === 'query' && (
          <div className="sidebar-section">
            <div className="field">
              <label className="field-label">Top-K &nbsp;<span className="mono text-cyan">{topK}</span></label>
              <input
                type="range" min="1" max="20" value={topK}
                onChange={e => setTopK(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--saffron)' }}
              />
            </div>
          </div>
        )}

        {/* Advanced settings */}
        {mode === 'query' && (
          <div className="sidebar-section sidebar-section--adv">
            <button className="adv-toggle" onClick={() => setShowAdv(v => !v)}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-1)' }}>Advanced</span>
              {showAdv ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>

            {showAdv && (
              <div className="gap-12" style={{ marginTop: 12 }}>
                <div className="toggle-row">
                  <span style={{ fontSize: '0.76rem', color: 'var(--text-1)' }}>CFM Bridge</span>
                  <label className="switch">
                    <input type="checkbox" checked={bridge} onChange={e => setBridge(e.target.checked)} />
                    <span className="switch-thumb" />
                  </label>
                </div>
                <div className="field">
                  <label className="field-label">
                    ODE Steps: <span className="mono text-cyan">{odeSteps}</span>
                    &nbsp;
                    <span className="mono" style={{ fontSize: '0.62rem', color: 'var(--text-2)' }}>
                      ~{odeLatencyEstimate(odeSteps)} ms est.
                    </span>
                  </label>
                  <input
                    type="range" min="1" max="15" value={odeSteps}
                    onChange={e => setOdeSteps(parseInt(e.target.value))}
                    style={{ width: '100%', accentColor: 'var(--cyan)' }}
                  />
                  <div className="ode-range-labels">
                    <span>Fast (1)</span><span>Balanced (5)</span><span>Precise (15)</span>
                  </div>
                </div>
                <div className="toggle-row">
                  <span style={{ fontSize: '0.76rem', color: 'var(--text-1)' }}>Jaccard Reranking</span>
                  <label className="switch">
                    <input type="checkbox" checked={rerank} onChange={e => setRerank(e.target.checked)} />
                    <span className="switch-thumb" />
                  </label>
                </div>
              </div>
            )}
          </div>
        )}
      </aside>

      {/* ── MAIN CANVAS ──────────────────────── */}
      <section className="canvas">
        {mode === 'query'
          ? <QueryMode    params={params} onResult={onQuery} onCompare={onCompare} onHistoryAdd={addToHistory} />
          : <AblationMode params={params} />}

        {/* Query history at the bottom */}
        {mode === 'query' && (
          <HistoryPanel history={history} onReplay={replayQuery} />
        )}
      </section>
    </div>
  );
}
