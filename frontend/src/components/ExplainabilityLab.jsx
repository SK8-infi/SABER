import React from 'react';
import { Layers, Zap, Cpu, Activity, Info } from 'lucide-react';

export default function ExplainabilityLab() {
  const bandsS2 = [
    { name: 'B1 (Coastal Aerosol)', wave: '0.443 μm', desc: 'Atmospheric correction, ocean color' },
    { name: 'B2 (Blue)', wave: '0.490 μm', desc: 'Soil/vegetation discrimination' },
    { name: 'B3 (Green)', wave: '0.560 μm', desc: 'Peak vegetation reflection' },
    { name: 'B4 (Red)', wave: '0.665 μm', desc: 'Chlorophyll absorption peak' },
    { name: 'B5 (Red Edge 1)', wave: '0.705 μm', desc: 'Vegetation state assessment' },
    { name: 'B6 (Red Edge 2)', wave: '0.740 μm', desc: 'Leaf area index (LAI)' },
    { name: 'B7 (Red Edge 3)', wave: '0.783 μm', desc: 'Canopy structure' },
    { name: 'B8 (NIR)', wave: '0.842 μm', desc: 'Biomass & water body boundary' },
    { name: 'B8A (Narrow NIR)', wave: '0.865 μm', desc: 'Vegetation moisture' },
    { name: 'B9 (Water Vapor)', wave: '0.945 μm', desc: 'Atmospheric vapor absorption' },
    { name: 'B11 (SWIR 1)', wave: '1.610 μm', desc: 'Snow/cloud discrimination, moisture' },
    { name: 'B12 (SWIR 2)', wave: '2.190 μm', desc: 'Geological mapping, soil composition' }
  ];

  const bandsS1 = [
    { name: 'C-Band VV Polarization', wave: '5.405 μm (5.55 cm)', desc: 'Vertical Transmit / Vertical Receive. Sensitive to surface roughness and soil moisture.' },
    { name: 'C-Band VH Polarization', wave: '5.405 μm (5.55 cm)', desc: 'Vertical Transmit / Horizontal Receive. Sensitive to volumetric vegetation canopy scattering.' }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      
      <div className="scientific-card">
        <div className="card-header">
          <span className="card-title">
            <Layers className="card-title-icon" size={16} /> Physical Sensing & Wavelength Explainability Lab
          </span>
          <span className="mono-tag saffron">ISRO BAH 2026 EXPLAINABILITY</span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Explains why Sentinel-1 SAR and Sentinel-2 Optical sensors register different physical surface properties and how DOFA's 
          wavelength hypernetwork generates dynamic patch projection weights W_proj(lambda) = g_hyper(lambda_c).
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        
        {/* Optical Sentinel-2 Spectral Profile */}
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title" style={{ color: 'var(--accent-cyan)' }}>
              Sentinel-2 Optical Multispectral (12 Bands)
            </span>
            <span className="mono-tag cyan">SOLAR REFLECTANCE</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '420px', overflowY: 'auto' }}>
            {bandsS2.map((b, idx) => (
              <div key={idx} style={{ backgroundColor: 'var(--bg-dark)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>{b.name}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{b.desc}</div>
                </div>
                <div className="data-mono" style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>
                  {b.wave}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* SAR Sentinel-1 Microwave Profile */}
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title" style={{ color: 'var(--accent-saffron)' }}>
              Sentinel-1 SAR C-Band Radar (2 Channels)
            </span>
            <span className="mono-tag saffron">MICROWAVE BACKSCATTER</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {bandsS1.map((b, idx) => (
              <div key={idx} style={{ backgroundColor: 'var(--bg-dark)', padding: '14px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-saffron)' }}>{b.name}</div>
                  <div className="data-mono" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{b.wave}</div>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                  {b.desc}
                </p>
              </div>
            ))}

            <div style={{ backgroundColor: 'rgba(255, 153, 51, 0.08)', border: '1px dashed var(--accent-saffron)', padding: '14px', borderRadius: '6px', marginTop: '12px' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-saffron)', marginBottom: '4px' }}>
                🔬 DOFA Hypernetwork Mechanics
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                Instead of using static RGB filters, DOFA uses a 2-layer MLP hypernetwork that accepts central wavelengths $\lambda_c$ 
                (in micrometers) and outputs 768-dimensional patch projection weights for each band dynamically. This prevents spectral distortion across multi-sensor imagery.
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
