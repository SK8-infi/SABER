import React, { useState, useEffect } from 'react';
import { Activity, Compass, Info } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function EmbeddingSpaceInspector() {
  const [pointsData, setPointsData] = useState(null);

  useEffect(() => {
    fetch('/api/embedding/points')
      .then((res) => res.json())
      .then((data) => setPointsData(data))
      .catch((err) => console.error("Error loading embedding points:", err));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      
      <div className="scientific-card">
        <div className="card-header">
          <span className="card-title">
            <Activity className="card-title-icon" size={16} /> 2D Shared Embedding Topology & Trajectory Inspector (UMAP / PCA Projection)
          </span>
          <span className="mono-tag cyan">SHARED EMBEDDING HYPERSPHERE</span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Visualizes the shared 768-dimensional latent space manifold reduced to 2D coordinates via UMAP. Demonstrates how the 
          <strong>Conditional Flow Matching Latent Bridge</strong> transports Sentinel-1 SAR source points (blue cluster) directly onto the Sentinel-2 Optical hypersphere (green cluster).
        </p>
      </div>

      {pointsData && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '20px' }}>
          
          {/* Scatter Plot Chart */}
          <div className="scientific-card" style={{ height: '440px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <XAxis type="number" dataKey="x" name="UMAP-1" stroke="#64748B" tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }} />
                <YAxis type="number" dataKey="y" name="UMAP-2" stroke="#64748B" tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#121721', border: '1px solid #1E2638', fontFamily: 'var(--font-mono)', fontSize: '12px' }} />
                
                {/* S1 Cluster */}
                <Scatter name="Sentinel-1 SAR Source" data={pointsData.s1_cluster} fill="#3B82F6" opacity={0.6} />
                
                {/* S2 Cluster */}
                <Scatter name="Sentinel-2 MS Target" data={pointsData.s2_cluster} fill="#10B981" opacity={0.6} />

                {/* SABER Transformed Cluster */}
                <Scatter name="SABER Transformed (CFM)" data={pointsData.bridged_cluster} fill="#FF9933" opacity={0.9} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Legend & ODE Trajectory Breakdown */}
          <div className="scientific-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div className="card-header">
              <span className="card-title" style={{ fontSize: '0.85rem' }}>Cluster Legend</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.78rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#3B82F6' }} />
                <span>Sentinel-1 SAR Source (z1)</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#10B981' }} />
                <span>Sentinel-2 Optical Target (z2)</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#FF9933' }} />
                <span>SABER Transformed (z_1-&gt;2)</span>
              </div>
            </div>

            <div className="card-header" style={{ marginTop: '8px' }}>
              <span className="card-title" style={{ fontSize: '0.85rem' }}>5-Step ODE Solver Path</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.73rem', fontFamily: 'var(--font-mono)' }}>
              {pointsData.trajectory.map((step) => (
                <div key={step.step} style={{ display: 'flex', justifyContent: 'space-between', backgroundColor: 'var(--bg-dark)', padding: '6px 10px', borderRadius: '4px' }}>
                  <span>tau = {step.tau.toFixed(2)}</span>
                  <span style={{ color: 'var(--accent-saffron)' }}>({step.x}, {step.y})</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}

    </div>
  );
}
