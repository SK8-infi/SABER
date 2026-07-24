import React from 'react';
import { ShieldCheck, Zap, Layers, Cpu, Compass, AlertCircle, ArrowRight } from 'lucide-react';

export default function OverviewSection({ onStartDemo }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', width: '100%' }}>
      {/* Hero Banner */}
      <div className="scientific-card" style={{ background: 'linear-gradient(135deg, rgba(18, 23, 33, 0.95) 0%, rgba(30, 38, 56, 0.6) 100%)', border: '1px solid rgba(255, 153, 51, 0.3)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span className="mono-tag saffron">ISRO BAH 2026 GRAND FINALE</span>
              <span className="mono-tag cyan">PROBLEM STATEMENT 11</span>
              <span className="mono-tag green">TEAM SENTINEL8</span>
            </div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.02em', marginBottom: '12px' }}>
              SABER: Sensor-Agnostic Bridged Embedding Retrieval
            </h1>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '900px', fontSize: '0.95rem', lineHeight: '1.6' }}>
              A scientifically rigorous cross-modal satellite image retrieval framework. SABER maps disparate Earth observation modalities 
              (Synthetic Aperture Radar & Multispectral Optical) into a unified, metric-optimized hypersphere via wavelength hypernetworks, 
              LoRA adapters, and generative Conditional Flow Matching (CFM) ODE latent bridges.
            </p>
          </div>
          <button className="primary-action-btn" onClick={onStartDemo} style={{ padding: '12px 24px', fontSize: '0.9rem' }}>
            LAUNCH LIVE RETRIEVAL WORKSPACE <ArrowRight size={16} />
          </button>
        </div>
      </div>

      {/* Narrative Arc 3-Column Comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>
        
        {/* Card 1: Problem */}
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title" style={{ color: 'var(--accent-red)' }}>
              <AlertCircle size={16} /> 1. Physical Sensing Gap
            </span>
            <span className="mono-tag">PROBLEM</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '12px' }}>
            Satellite Earth observation relies on heterogeneous sensors. <strong>Sentinel-1 SAR</strong> uses active C-band microwave radar 
            ($\lambda = 5.405\,\mu\text{m}$) measuring dielectric roughness, while <strong>Sentinel-2 Optical</strong> captures 12-band passive 
            solar reflectance ($\lambda \in [0.443, 2.190]\,\mu\text{m}$).
          </p>
          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '10px', borderRadius: '6px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
            x_SAR in R^[2 x H x W]  vs  x_MS in R^[12 x H x W]
          </div>
        </div>

        {/* Card 2: Limitation */}
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title" style={{ color: 'var(--accent-saffron)' }}>
              <Layers size={16} /> 2. Modality Collapse
            </span>
            <span className="mono-tag">LIMITATION</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '12px' }}>
            Conventional joint training forces models into joint retrain overhead or results in Severe Modality Collapse. Standard linear 
            projection layers fail to bridge the complex non-linear probability shift between microwave structural geometry and optical spectral signatures.
          </p>
          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '10px', borderRadius: '6px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-red)' }}>
            Baseline Cross-Modal mAP: 71.95% (20% Domain Shift Deficit)
          </div>
        </div>

        {/* Card 3: SABER Solution */}
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title" style={{ color: 'var(--accent-green)' }}>
              <ShieldCheck size={16} /> 3. SABER Latent Bridge
            </span>
            <span className="mono-tag green">CONTRIBUTION</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '12px' }}>
            SABER introduces a <strong>Conditional Flow Matching (CFM) ODE latent bridge</strong> that conditionally transports SAR query descriptors 
            across the probability boundary to the optical hypersphere in 5 Euler steps, closing 67% of the cross-modal retrieval gap.
          </p>
          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '10px', borderRadius: '6px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>
            SABER Cross-Modal mAP: 83.23% (+11.28 pp Boost)
          </div>
        </div>

      </div>

      {/* Core Scientific Pillars Summary */}
      <div className="scientific-card">
        <div className="card-header">
          <span className="card-title">
            <Cpu className="card-title-icon" size={16} /> 4 Foundational Scientific Architecture Pillars
          </span>
          <span className="mono-tag cyan">SABER DESIGN</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-saffron)', marginBottom: '4px' }}>
              1. DOFA ViT Backbone
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Wavelength Hypernetwork dynamically conditions patch weights using central wavelengths ($\lambda_c$) of active bands.
            </p>
          </div>

          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: '4px' }}>
              2. PEFT LoRA Adapters
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Adapts QKV & MLP projections ($r=16, \alpha=32$). Freezes 99.74% of ViT parameters for ultra-low memory training.
            </p>
          </div>

          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-green)', marginBottom: '4px' }}>
              3. CFM Latent Bridge
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Solves continuous ODE $\text{d}z/\text{d}\tau = v(z, \tau)$ in 5 GPU Euler steps to map source $z_1$ to target hypersphere $z_2$.
            </p>
          </div>

          <div style={{ backgroundColor: 'var(--bg-dark)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-blue)', marginBottom: '4px' }}>
              4. VICReg + Jaccard Loss
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Directly regresses cosine similarity against BigEarthNet 19 multi-label class overlap indices with neighborhood constraints.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
