import { useState } from 'react';
import { AlertCircle, Layers, ShieldCheck, Cpu, ArrowRight } from 'lucide-react';
import SponsorsSection from './SponsorsSection';


const NODES = [
  {
    id: 'input',  tag: 'INPUT',    color: 'var(--blue)',
    title: 'Query Scene',
    specs: '[B, C, 120, 120] — C=2 (SAR) or C=12 (Optical)',
    purpose: 'Accepts heterogeneous satellite image rasters — Sentinel-1 VV/VH radar or Sentinel-2 12-band multispectral.',
    math: 'x ∈ ℝ^[C×H×W]',
    inputs: 'Raw sensor rasters',
    outputs: 'Float32 tensor',
  },
  {
    id: 'adapter', tag: 'CONV1×1', color: 'var(--cyan)',
    title: 'Input Adapter',
    specs: 'Conv1×1 — C → 3 channels',
    purpose: 'Adapts arbitrary spectral channel counts to 3-channel visual tokens while preserving energy distributions.',
    math: 'x_proj = Conv1×1(x)',
    inputs: '[B, C, 120, 120]',
    outputs: '[B, 3, 120, 120]',
  },
  {
    id: 'dofa',   tag: 'HYPERNET', color: 'var(--saffron)',
    title: 'DOFA ViT',
    specs: 'ViT-Base/16 — 111.3M — λ_c conditioned',
    purpose: 'Dynamically computes patch embedding weights tailored to exact spectral wavelengths of active bands.',
    math: 'W_proj(λ) = g_hyper(λ_c)',
    inputs: '[B, 3, 120, 120] + λ_c',
    outputs: 'Token sequence [B, 768]',
  },
  {
    id: 'lora',   tag: 'LoRA r=16', color: 'var(--green)',
    title: 'PEFT LoRA',
    specs: 'qkv, fc1, fc2 — 294.9K trainable (0.26%)',
    purpose: 'Fine-tunes attention projections without destroying pre-trained foundation knowledge.',
    math: 'W = W₀ + (α/r)·A·B',
    inputs: 'DOFA QKV projections',
    outputs: 'Adapted Transformer repr.',
  },
  {
    id: 'proj',   tag: 'MLP HEAD', color: 'var(--purple)',
    title: 'Projection Head',
    specs: '768→768→768 — GELU + Residual + L2-norm',
    purpose: 'Projects transformer features into a 768-dim L2-normalised embedding hypersphere.',
    math: 'z₁ = L2-norm(MLP(ViT(x₁))) ∈ ℝ^768',
    inputs: '[B, 768] ViT features',
    outputs: '[B, 768] latent z₁',
  },
  {
    id: 'bridge', tag: 'CFM ODE',  color: 'var(--saffron)',
    title: 'CFM Bridge',
    specs: '5-step Euler — v_θ(z, τ; z_q)',
    purpose: 'Generatively transports z₁ (SAR) to optical hypersphere z₂ by solving a continuous ODE.',
    math: 'dz/dτ = v_θ(z, τ; z_q),  τ ∈ [0, 1]',
    inputs: 'Source latent z₁',
    outputs: 'Aligned latent z₁→₂',
  },
  {
    id: 'faiss',  tag: 'FAISS IP', color: 'var(--red)',
    title: 'FAISS Engine',
    specs: 'IndexFlatIP — 11,866 scenes',
    purpose: 'Sub-millisecond inner product search against pre-indexed gallery embeddings.',
    math: 'k* = argmax_k ⟨z₁→₂, z₂,k⟩',
    inputs: 'Query vector [768]',
    outputs: 'Top-K indices & scores',
  },
];

export default function AboutPanel() {
  const [sel, setSel] = useState(null);

  return (
    <div className="gap-20">

      {/* Hero */}
      <div className="card" style={{ borderColor: 'rgba(255,153,51,0.2)' }}>
        <div className="about-hero">
          <div className="gap-12">
            <div className="inline-row">
              <span className="tag tag--saffron">ISRO BAH 2026</span>
              <span className="tag tag--cyan">PS-11</span>
              <span className="tag tag--green">Team Sentinel8</span>
            </div>
            <h1 style={{ fontSize: '1.3rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
              SABER — Sensor-Agnostic Bridged Embedding Retrieval
            </h1>
            <p className="section-sub" style={{ maxWidth: 760 }}>
              Cross-modal satellite image retrieval. SAR &amp; Multispectral EO modalities unified onto
              a metric-optimised hypersphere via wavelength hypernetworks, LoRA adapters, and
              Conditional Flow Matching ODE latent bridges.
            </p>
          </div>
        </div>
      </div>

      {/* 3-col problem / limitation / contribution */}
      <div className="overview-grid">
        <div className="card">
          <div className="card-head">
            <span className="card-title text-red"><AlertCircle size={13} /> Sensing Gap</span>
            <span className="tag">PROBLEM</span>
          </div>
          <p className="section-sub" style={{ marginBottom: 10 }}>
            Sentinel-1 SAR uses active C-band microwave (λ=5.405 µm) while Sentinel-2 captures 12-band
            passive solar reflectance (λ ∈ [0.443, 2.190] µm). Fundamentally different physics.
          </p>
          <div className="code-block">x_SAR ∈ ℝ^[2×H×W]  vs  x_MS ∈ ℝ^[12×H×W]</div>
        </div>

        <div className="card">
          <div className="card-head">
            <span className="card-title text-saffron"><Layers size={13} /> Modality Collapse</span>
            <span className="tag">LIMITATION</span>
          </div>
          <p className="section-sub" style={{ marginBottom: 10 }}>
            Conventional joint training causes modality collapse — linear projections cannot bridge the
            non-linear probability shift between microwave geometry and optical spectral signatures.
          </p>
          <div className="code-block" style={{ color: 'var(--red)' }}>
            Baseline Cross-Modal mAP: 71.95%
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <span className="card-title text-green"><ShieldCheck size={13} /> CFM Bridge</span>
            <span className="tag tag--green">CONTRIBUTION</span>
          </div>
          <p className="section-sub" style={{ marginBottom: 10 }}>
            CFM ODE latent bridge conditionally transports SAR descriptors to the optical hypersphere in
            5 Euler steps, closing 67% of the cross-modal gap.
          </p>
          <div className="code-block" style={{ color: 'var(--green)' }}>
            SABER Cross-Modal mAP: 83.23% — +11.28 pp
          </div>
        </div>
      </div>

      {/* Architecture pillars */}
      <div className="card">
        <div className="card-head">
          <span className="card-title"><Cpu size={13} /> Architecture Pillars</span>
          <span className="tag tag--cyan">SABER DESIGN</span>
        </div>
        <div className="overview-pillars">
          {[
            { color: 'var(--saffron)', title: '1. DOFA ViT',         body: 'Wavelength Hypernetwork dynamically conditions patch weights via λ_c of active bands.' },
            { color: 'var(--cyan)',    title: '2. PEFT LoRA r=16',   body: 'Adapts QKV & MLP projections. Freezes 99.74% of ViT params — ultra-low memory.' },
            { color: 'var(--green)',   title: '3. CFM Latent Bridge', body: 'Solves ODE dz/dτ = v(z,τ) in 5 GPU Euler steps to map z₁ to target hypersphere z₂.' },
            { color: 'var(--blue)',    title: '4. VICReg + Jaccard',  body: 'Directly regresses cosine similarity against BigEarthNet 19 multi-label Jaccard overlap.' },
          ].map(p => (
            <div className="pillar" key={p.title}>
              <div style={{ fontSize: '0.72rem', fontWeight: 700, color: p.color, marginBottom: 5 }}>{p.title}</div>
              <p className="section-sub" style={{ fontSize: '0.7rem' }}>{p.body}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive pipeline */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Interactive Pipeline</span>
          <span className="tag">CLICK A MODULE</span>
        </div>

        <div style={{ overflowX: 'auto', paddingBottom: 4 }}>
          <div className="pipeline-flow">
            {NODES.map((n, i) => (
              <div key={n.id} className="pipeline-step">
                <div
                  className={`pipe-node${sel?.id === n.id ? ' active' : ''}`}
                  style={{ borderColor: sel?.id === n.id ? n.color : undefined }}
                  onClick={() => setSel(sel?.id === n.id ? null : n)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                    <span className="tag" style={{ color: n.color, borderColor: `${n.color}44`, fontSize: '0.58rem' }}>{n.tag}</span>
                    <span className="mono text-dim" style={{ fontSize: '0.58rem' }}>0{i + 1}</span>
                  </div>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, marginBottom: 3 }}>{n.title}</div>
                  <div className="mono text-dim" style={{ fontSize: '0.6rem', overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                    {n.specs.split('—')[0].trim()}
                  </div>
                </div>
                {i < NODES.length - 1 && (
                  <ArrowRight size={13} color="var(--text-3)" style={{ flexShrink: 0, margin: '0 2px' }} />
                )}
              </div>
            ))}
          </div>
        </div>

        {sel ? (
          <div className="pipe-detail" style={{ borderLeftColor: sel.color }}>
            <div className="card-head">
              <span className="card-title" style={{ color: sel.color }}>{sel.title}</span>
              <span className="tag" style={{ color: sel.color, borderColor: `${sel.color}44` }}>{sel.tag}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 16 }}>
              <div className="gap-12">
                <div>
                  <div className="field-label" style={{ marginBottom: 5 }}>Scientific Purpose</div>
                  <p className="section-sub">{sel.purpose}</p>
                </div>
                <div>
                  <div className="field-label" style={{ marginBottom: 5 }}>Math</div>
                  <div className="code-block">{sel.math}</div>
                </div>
              </div>
              <div className="metric-box gap-12">
                <div className="gap-4">
                  <span className="metric-label">Config</span>
                  <span className="mono" style={{ fontSize: '0.73rem', color: 'var(--text-1)' }}>{sel.specs}</span>
                </div>
                <div className="divider" />
                <div className="gap-4">
                  <span className="metric-label">Input</span>
                  <span className="mono text-saffron" style={{ fontSize: '0.73rem' }}>{sel.inputs}</span>
                </div>
                <div className="gap-4">
                  <span className="metric-label">Output</span>
                  <span className="mono text-green" style={{ fontSize: '0.73rem' }}>{sel.outputs}</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '16px 0', color: 'var(--text-3)', fontSize: '0.78rem' }}>
            Click a block above to inspect details
          </div>
        )}
      </div>

      {/* Supporters & Sponsors */}
      <SponsorsSection />

    </div>
  );
}

