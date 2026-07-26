import { useState, useEffect, useCallback } from 'react';
import {
  Search, RefreshCw, ChevronDown, ChevronUp, Eye,
  GitCompare, Zap, AlertCircle, ShieldCheck, ArrowRight,
} from 'lucide-react';

/* ─── small sub-components ─────────────────────────── */

function LatencyStrip({ lats }) {
  const items = [
    { label: 'Feat Ext', val: lats.feature_extraction_ms,  color: 'var(--cyan)' },
    { label: 'CFM ODE',  val: lats.latent_bridge_ms,        color: 'var(--saffron)' },
    { label: 'FAISS',    val: lats.faiss_search_ms,         color: 'var(--blue)' },
    { label: 'Total',    val: lats.total_latency_ms,        color: 'var(--green)', bold: true },
  ];
  return (
    <div className="lat-strip">
      {items.map(m => (
        <div key={m.label} className={`lat-item${m.bold ? ' lat-item--total' : ''}`}>
          <span className="metric-label">{m.label}</span>
          <span className="mono" style={{ fontSize: '0.82rem', fontWeight: m.bold ? 700 : 500, color: m.color }}>
            {m.val} ms
          </span>
        </div>
      ))}
    </div>
  );
}

function SimilarityBar({ pct, color = 'var(--saffron)' }) {
  return (
    <div className="sim-bar-track">
      <div className="sim-bar-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

function CandidateCard({ c, query, onCompare, rank }) {
  const hue = c.similarity_score >= 80 ? 'var(--green)'
    : c.similarity_score >= 60 ? 'var(--saffron)'
    : 'var(--red)';

  return (
    <div className="cand-card">
      <span className="cand-rank">#{rank}</span>
      <span className="cand-score" style={{ background: hue === 'var(--green)' ? 'var(--green)' : hue === 'var(--saffron)' ? 'var(--saffron)' : 'var(--red)' }}>
        {c.similarity_score}%
      </span>
      <img src={c.thumbnail} alt={c.name} className="cand-thumb" />
      <div className="cand-body">
        <div className="cand-name">{c.name}</div>
        <SimilarityBar pct={c.similarity_score} color={hue} />
        <div className="list-row" style={{ marginTop: 2 }}>
          <span className="text-dim" style={{ fontSize: '0.7rem' }}>Jaccard</span>
          <span className="mono text-cyan" style={{ fontSize: '0.7rem' }}>{c.jaccard_overlap}%</span>
        </div>
        <div className="chips" style={{ marginTop: 2 }}>
          {c.active_classes.slice(0, 3).map((cl, i) => (
            <span className="chip" key={i}>{cl}</span>
          ))}
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

/* ─── QUERY MODE ────────────────────────────────────── */

function QueryMode({ params, onResult, onCompare }) {
  const { dataset, srcMod, tgtMod, qIdx, topK, bridge, rerank, odeSteps } = params;
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/retrieval/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name:     dataset,
          query_index:      qIdx,
          source_modality:  srcMod,
          target_modality:  tgtMod,
          top_k:            topK,
          enable_bridge:    bridge,
          enable_rerank:    rerank,
          ode_steps:        odeSteps,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
      if (onResult) onResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [dataset, qIdx, srcMod, tgtMod, topK, bridge, rerank, odeSteps]);

  // auto-run when core params change
  useEffect(() => { run(); }, [dataset, qIdx, srcMod, tgtMod, topK]);

  return (
    <div className="results-area">
      {loading && (
        <div className="loading-state">
          <div className="spinner" />
          <span className="text-dim">Running retrieval…</span>
        </div>
      )}

      {error && !loading && (
        <div className="error-state">
          <AlertCircle size={16} color="var(--red)" />
          <span style={{ color: 'var(--red)', fontSize: '0.82rem' }}>Query failed: {error}</span>
        </div>
      )}

      {result && !loading && (
        <>
          {/* Query header */}
          <div className="query-header">
            <div className="query-img-wrap">
              <img src={result.query.thumbnail} alt="Query" className="query-thumb" />
              <span className="query-mod-badge">{result.query.source_modality.toUpperCase()}</span>
            </div>
            <div className="query-meta">
              <div className="query-name">{result.query.name}</div>
              <div className="chips" style={{ marginTop: 4 }}>
                {result.query.active_classes.slice(0, 6).map((cl, i) => (
                  <span className="chip chip--query" key={i}>{cl}</span>
                ))}
              </div>
              <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-2)' }}>
                Gallery: <span className="text-dim">{result.query.target_modality.toUpperCase()}</span>
                &nbsp;·&nbsp;Top-{topK}
                &nbsp;·&nbsp;Bridge: <span style={{ color: bridge ? 'var(--green)' : 'var(--red)' }}>{bridge ? 'ON' : 'OFF'}</span>
              </div>
            </div>
            <LatencyStrip lats={result.latency_telemetry} />
          </div>

          {/* Candidates */}
          <div className="cand-grid">
            {result.candidates.map(c => (
              <CandidateCard key={c.rank} c={c} query={result.query} onCompare={onCompare} rank={c.rank} />
            ))}
          </div>
        </>
      )}

      {!result && !loading && !error && (
        <div className="empty-state">
          <Search size={32} color="var(--border-2)" />
          <span className="text-dim">Configure params and hit Execute</span>
        </div>
      )}
    </div>
  );
}

/* ─── ABLATION MODE ─────────────────────────────────── */

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
        <div className="loading-state">
          <div className="spinner" />
          <span className="text-dim">Running ablation…</span>
        </div>
      )}

      {error && !loading && (
        <div className="error-state">
          <AlertCircle size={16} color="var(--red)" />
          <span style={{ color: 'var(--red)', fontSize: '0.82rem' }}>Ablation failed: {error}</span>
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
              <span className="metric-label">Jaccard Δ</span>
              <span className="mono text-saffron" style={{ fontSize: '1rem', fontWeight: 700 }}>
                +{data.delta.jaccard_improvement}%
              </span>
            </div>
          </div>

          {/* Side-by-side */}
          <div className="ablation-cols">
            {/* Bridge OFF */}
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
                  <div key={c.rank} className="abl-row">
                    <span className="mono text-dim abl-rank">#{c.rank}</span>
                    <span className="abl-row-name">{c.name}</span>
                    <SimilarityBar pct={c.similarity_score} color="var(--red)" />
                    <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-1)', flexShrink: 0 }}>{c.similarity_score}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Bridge ON */}
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
                  <div key={c.rank} className="abl-row">
                    <span className="mono text-green abl-rank" style={{ fontWeight: 700 }}>#{c.rank}</span>
                    <span className="abl-row-name">{c.name}</span>
                    <SimilarityBar pct={c.similarity_score} color="var(--green)" />
                    <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--green)', fontWeight: 700, flexShrink: 0 }}>{c.similarity_score}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/* ─── MAIN COMPONENT ────────────────────────────────── */

export default function RetrievalWorkspace({ onQuery, onCompare }) {
  const [mode, setMode]         = useState('query');   // 'query' | 'ablation'
  const [dataset, setDataset]   = useState('ben14k');
  const [srcMod, setSrcMod]     = useState('s1');
  const [tgtMod, setTgtMod]     = useState('s2');
  const [qIdx, setQIdx]         = useState(0);
  const [topK, setTopK]         = useState(5);
  const [showAdv, setShowAdv]   = useState(false);
  const [bridge, setBridge]     = useState(true);
  const [rerank, setRerank]     = useState(false);
  const [odeSteps, setOdeSteps] = useState(5);
  const isBen = dataset === 'ben14k';

  const handleDatasetChange = v => {
    setDataset(v);
    if (v === 'dsrsid') { setSrcMod('pan'); setTgtMod('ms'); }
    else                { setSrcMod('s1');  setTgtMod('s2'); }
  };

  const randomize = () => setQIdx(Math.floor(Math.random() * 2000));

  const params = { dataset, srcMod, tgtMod, qIdx, topK, bridge, rerank, odeSteps };

  return (
    <div className="workspace">

      {/* ── LEFT SIDEBAR ─────────────────────── */}
      <aside className="sidebar">

        {/* Mode toggle */}
        <div className="mode-toggle">
          <button
            className={`mode-btn${mode === 'query' ? ' mode-btn--active' : ''}`}
            onClick={() => setMode('query')}
          >
            <Search size={12} /> Query
          </button>
          <button
            className={`mode-btn${mode === 'ablation' ? ' mode-btn--active' : ''}`}
            onClick={() => setMode('ablation')}
          >
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

        {/* Modality (hidden in ablation — locked to cross-modal) */}
        {mode === 'query' && (
          <div className="sidebar-section">
            <div className="field">
              <label className="field-label">Source</label>
              <select className="select" value={srcMod} onChange={e => setSrcMod(e.target.value)}>
                {isBen
                  ? <><option value="s1">Sentinel-1 SAR (2ch)</option><option value="s2">Sentinel-2 MS (12ch)</option></>
                  : <><option value="pan">Gaofen-1 PAN (1ch)</option><option value="ms">Gaofen-1 MS (4ch)</option></>}
              </select>
            </div>
            <div className="field" style={{ marginTop: 10 }}>
              <label className="field-label">Target Gallery</label>
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

        {/* Query index */}
        <div className="sidebar-section">
          <div className="field">
            <label className="field-label">
              Scene Index&nbsp;
              <span className="mono text-saffron">#{qIdx}</span>
            </label>
            <input
              type="number" className="input" min="0" max="2965"
              value={qIdx}
              onChange={e => setQIdx(parseInt(e.target.value) || 0)}
            />
          </div>
          <button className="btn btn--ghost btn--sm btn--full" style={{ marginTop: 8 }} onClick={randomize}>
            <RefreshCw size={11} /> Random Scene
          </button>
        </div>

        {/* Top-K (query mode only) */}
        {mode === 'query' && (
          <div className="sidebar-section">
            <div className="field">
              <label className="field-label">
                Top-K &nbsp;<span className="mono text-cyan">{topK}</span>
              </label>
              <input
                type="range" min="1" max="20" value={topK}
                onChange={e => setTopK(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--saffron)' }}
              />
            </div>
          </div>
        )}

        {/* Advanced (query mode only) */}
        {mode === 'query' && (
          <div className="sidebar-section sidebar-section--adv">
            <button
              className="adv-toggle"
              onClick={() => setShowAdv(v => !v)}
            >
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
                  <label className="field-label">ODE Steps: {odeSteps}</label>
                  <input
                    type="range" min="1" max="15" value={odeSteps}
                    onChange={e => setOdeSteps(parseInt(e.target.value))}
                    style={{ width: '100%', accentColor: 'var(--cyan)' }}
                  />
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
          ? <QueryMode    params={params} onResult={onQuery} onCompare={onCompare} />
          : <AblationMode params={params} />}
      </section>
    </div>
  );
}
