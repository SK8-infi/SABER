import React, { useState } from 'react';
import { Cpu, ArrowRight, Layers, Sliders, Zap, Database, CheckCircle, Info } from 'lucide-react';

export default function ArchitectureFlow({ activeExecution }) {
  const [selectedNode, setSelectedNode] = useState(null);

  const nodes = [
    {
      id: 'input',
      title: 'Query Satellite Scene',
      tag: 'INPUT TENSOR',
      color: '#3B82F6',
      specs: 'Dimensions: [B, C, 120, 120] | C=2 (SAR) or C=12 (Optical)',
      purpose: 'Accepts heterogeneous satellite image rasters (Sentinel-1 VV/VH radar or Sentinel-2 12 multispectral bands).',
      math: 'x in R^[C x H x W]',
      inputs: 'Raw sensor data rasters / numpy arrays',
      outputs: 'Normalized Float32 tensor'
    },
    {
      id: 'adapter',
      title: 'Input Adapter Layer',
      tag: 'CONV1X1 / PROJ',
      color: '#00E5FF',
      specs: 'Conv1x1 projection mapping C channels -> 3 channel ViT format',
      purpose: 'Adapts arbitrary spectral channel counts to 3-channel visual tokens while preserving energy distributions.',
      math: 'x_proj = Conv1x1(x)',
      inputs: '[B, C, 120, 120]',
      outputs: '[B, 3, 120, 120]'
    },
    {
      id: 'dofa',
      title: 'DOFA Foundation ViT',
      tag: 'WAVELENGTH HYPERNET',
      color: '#FF9933',
      specs: 'ViT-Base/16 (111.3M parameters) | Conditioned on central wavelengths lambda_c',
      purpose: 'Dynamically computes patch embedding projection weights tailored to the exact spectral wavelengths of active bands.',
      math: 'W_proj(lambda) = g_hyper(lambda_c)',
      inputs: '[B, 3, 120, 120] + lambda_c',
      outputs: 'Token sequence [B, 768]'
    },
    {
      id: 'lora',
      title: 'PEFT LoRA Adapters',
      tag: 'TRAINABLE (r=16)',
      color: '#10B981',
      specs: 'Target Modules: qkv, fc1, fc2 | Trainable parameters: 294.9K (0.26%)',
      purpose: 'Fine-tunes attention projections without destroying pre-trained foundation knowledge or causing representation collapse.',
      math: 'W = W_0 + (alpha / r) * A * B',
      inputs: 'DOFA Block Query/Key/Value projections',
      outputs: 'Adapted Transformer representation'
    },
    {
      id: 'proj_head',
      title: '3-Layer Projection Head',
      tag: 'MLP + LAYERNORM',
      color: '#8B5CF6',
      specs: '768 -> 768 -> 768 | GELU activation + Residual connection',
      purpose: 'Projects transformer features into a 768-dimensional L2-normalized embedding space optimized for cosine similarity.',
      math: 'z_1 = MLP(ViT(x_1)) in R^768',
      inputs: '[B, 768] ViT features',
      outputs: '[B, 768] Source Latent Vector z_1'
    },
    {
      id: 'bridge',
      title: 'CFM Latent Bridge',
      tag: '5-STEP EULER ODE',
      color: '#F59E0B',
      specs: 'Conditional Flow Matching Vector Field v_theta(z, tau; z_q)',
      purpose: 'Generatively transports source query vector z_1 (SAR) to target gallery hypersphere z_2 (Optical) solving continuous ODE.',
      math: 'dz / dtau = v_theta(z, tau; z_query), tau in [0, 1]',
      inputs: 'Source Latent z_1',
      outputs: 'Target-Aligned Latent Vector z_1->2'
    },
    {
      id: 'faiss',
      title: 'FAISS Flat IP Engine',
      tag: 'VECTOR DATABASE',
      color: '#EC4899',
      specs: 'IndexFlatIP | Cosine Inner Product | 11,866 Gallery Scenes',
      purpose: 'Performs sub-millisecond inner product vector search against pre-indexed gallery embeddings to retrieve Top-K items.',
      math: 'k* = argmax_k <z_1->2, z_2,k>',
      inputs: 'Query vector [768]',
      outputs: 'Top-K Ranked Candidate Indices & Cosine Scores'
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      {/* Header Info */}
      <div className="scientific-card">
        <div className="card-header">
          <span className="card-title">
            <Cpu className="card-title-icon" size={16} /> Interactive SABER Pipeline Architecture Visualization
          </span>
          <span className="mono-tag saffron">ISRO BAH 2026 SPECIFICATION</span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Click on any architectural module block below to inspect its scientific purpose, tensor dimensions, mathematical formulation, 
          and hyperparameter profile. When a query is executed, the tensor flow pulse animates live across the pipeline.
        </p>
      </div>

      {/* Interactive Node Graph Pipeline */}
      <div className="scientific-card" style={{ backgroundColor: 'var(--bg-dark)', padding: '24px', overflowX: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', minWidth: '1100px', gap: '12px' }}>
          {nodes.map((node, index) => {
            const isSelected = selectedNode?.id === node.id;
            return (
              <React.Fragment key={node.id}>
                {/* Node Box */}
                <div
                  onClick={() => setSelectedNode(node)}
                  style={{
                    flex: 1,
                    backgroundColor: isSelected ? 'var(--bg-card-hover)' : 'var(--bg-card)',
                    border: `1.5px solid ${isSelected ? node.color : 'var(--border-color)'}`,
                    borderRadius: '8px',
                    padding: '14px',
                    cursor: 'pointer',
                    position: 'relative',
                    boxShadow: isSelected ? `0 0 15px ${node.color}33` : 'none',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span className="mono-tag" style={{ color: node.color, borderColor: `${node.color}44`, fontSize: '0.65rem' }}>
                      {node.tag}
                    </span>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      0{index + 1}
                    </span>
                  </div>

                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                    {node.title}
                  </div>

                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {node.specs.split('|')[0]}
                  </div>
                </div>

                {/* Arrow Connector */}
                {index < nodes.length - 1 && (
                  <ArrowRight size={18} color="var(--text-muted)" style={{ flexShrink: 0 }} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Selected Node Deep Dive Drawer */}
      {selectedNode ? (
        <div className="scientific-card" style={{ borderLeft: `4px solid ${selectedNode.color}` }}>
          <div className="card-header">
            <span className="card-title" style={{ color: selectedNode.color }}>
              <Info size={16} /> Module Deep Dive: {selectedNode.title}
            </span>
            <span className="mono-tag" style={{ color: selectedNode.color, border: `1px solid ${selectedNode.color}44` }}>
              {selectedNode.tag}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
            <div>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', uppercase: true, marginBottom: '4px' }}>
                SCIENTIFIC PURPOSE
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px', lineHeight: '1.6' }}>
                {selectedNode.purpose}
              </p>

              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', uppercase: true, marginBottom: '4px' }}>
                MATHEMATICAL FORMULATION
              </div>
              <div style={{ backgroundColor: 'var(--bg-dark)', padding: '12px', borderRadius: '6px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--accent-cyan)', marginBottom: '16px' }}>
                {selectedNode.math}
              </div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-dark)', padding: '16px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px' }}>
                Module Specifications
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.78rem' }}>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Config & Hyperparameters: </span>
                  <span className="data-mono" style={{ color: 'var(--text-secondary)' }}>{selectedNode.specs}</span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Input Interface: </span>
                  <span className="data-mono" style={{ color: 'var(--accent-saffron)' }}>{selectedNode.inputs}</span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Output Interface: </span>
                  <span className="data-mono" style={{ color: 'var(--accent-green)' }}>{selectedNode.outputs}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          💡 Click on any pipeline block above to inspect scientific details.
        </div>
      )}
    </div>
  );
}
