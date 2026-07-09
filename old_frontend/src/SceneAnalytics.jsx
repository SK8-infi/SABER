import React from 'react';
import './index.css';
import { 
  Network,
  Activity,
  Layers,
  BarChart2,
  SlidersHorizontal,
  Info
} from 'lucide-react';

export default function SceneAnalytics() {
  return (
    <div className="dashboard-grid" style={{ gridTemplateColumns: '1fr 320px' }}>
      
      {/* LEFT: SPLIT SCREEN COMPARISON */}
      <div className="panel" style={{ height: 'calc(100vh - 180px)' }}>
        <div className="panel-header">
          <span>Cross-Modal Comparison Viewer</span>
          <SlidersHorizontal size={14} color="var(--text-tertiary)" />
        </div>
        <div className="panel-content" style={{ padding: '0', position: 'relative', overflow: 'hidden' }}>
          
          <div style={{ display: 'flex', height: '100%', width: '100%' }}>
            {/* Query Image (Left) */}
            <div style={{ flex: 1, position: 'relative', borderRight: '2px solid var(--accent-saffron)' }}>
              <img 
                src="/img_query.png" 
                alt="Query SAR" 
                style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
              />
              <div style={{ position: 'absolute', top: '24px', left: '24px', backgroundColor: 'rgba(10,10,11,0.8)', padding: '8px 12px', border: '1px solid var(--border-strong)', borderRadius: '2px' }}>
                <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', marginBottom: '4px' }}>SOURCE MODALITY</div>
                <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>SAR (Sentinel-1)</div>
              </div>
            </div>
            
            {/* Retrieved Image (Right) */}
            <div style={{ flex: 1, position: 'relative' }}>
              <img 
                src="/img_res1.png" 
                alt="Retrieved MS" 
                style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
              />
              <div style={{ position: 'absolute', top: '24px', right: '24px', backgroundColor: 'rgba(10,10,11,0.8)', padding: '8px 12px', border: '1px solid var(--border-strong)', borderRadius: '2px', textAlign: 'right' }}>
                <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', marginBottom: '4px' }}>TARGET MODALITY</div>
                <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>Multispectral (S2)</div>
              </div>
            </div>
          </div>

          {/* Slider Handle Mock */}
          <div style={{ 
            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
            width: '32px', height: '32px', backgroundColor: 'var(--accent-saffron)', borderRadius: '50%',
            display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'ew-resize',
            boxShadow: '0 0 0 4px rgba(229, 141, 87, 0.2)'
          }}>
            <div style={{ width: '2px', height: '12px', backgroundColor: '#000', margin: '0 2px' }}></div>
            <div style={{ width: '2px', height: '12px', backgroundColor: '#000', margin: '0 2px' }}></div>
          </div>
          
        </div>
      </div>

      {/* RIGHT: LATENT DIAGNOSTICS */}
      <div className="main-content">
        
        {/* Bridge Confidence */}
        <div className="panel" style={{ height: 'auto' }}>
          <div className="panel-header">
            <span>Bridge Confidence u(q)</span>
            <Activity size={14} color="var(--text-tertiary)" />
          </div>
          <div className="panel-content" style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ position: 'relative', width: '160px', height: '80px', overflow: 'hidden', marginBottom: '16px' }}>
              <div style={{ 
                width: '160px', height: '160px', borderRadius: '50%', 
                border: '12px solid var(--border-strong)', borderTopColor: 'var(--accent-saffron)', borderRightColor: 'var(--accent-saffron)',
                transform: 'rotate(-45deg)'
              }}></div>
              <div style={{ position: 'absolute', bottom: '0', left: '0', right: '0', textAlign: 'center' }}>
                <span style={{ fontSize: '1.5rem', fontFamily: 'JetBrains Mono', fontWeight: 700, color: 'var(--text-primary)' }}>0.94</span>
              </div>
            </div>
            <p style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', textAlign: 'center', lineHeight: 1.4 }}>
              Flow-matching predictor uncertainty score. High confidence indicates strong cross-modal structural alignment.
            </p>
          </div>
        </div>

        {/* Relevance Overlap */}
        <div className="panel" style={{ height: 'auto' }}>
          <div className="panel-header">
            <span>Relevance Overlap (Jaccard)</span>
            <BarChart2 size={14} color="var(--text-tertiary)" />
          </div>
          <div className="panel-content" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8125rem' }}>
                  <span style={{ color: 'var(--text-primary)' }}>Water Bodies</span>
                  <span className="data-mono" style={{ color: 'var(--text-secondary)' }}>98.2%</span>
                </div>
                <div style={{ height: '4px', backgroundColor: 'var(--border-strong)', borderRadius: '2px' }}>
                  <div style={{ width: '98.2%', height: '100%', backgroundColor: 'var(--accent-saffron)' }}></div>
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8125rem' }}>
                  <span style={{ color: 'var(--text-primary)' }}>Urban Fabric</span>
                  <span className="data-mono" style={{ color: 'var(--text-secondary)' }}>84.5%</span>
                </div>
                <div style={{ height: '4px', backgroundColor: 'var(--border-strong)', borderRadius: '2px' }}>
                  <div style={{ width: '84.5%', height: '100%', backgroundColor: 'var(--text-secondary)' }}></div>
                </div>
              </div>
              
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8125rem' }}>
                  <span style={{ color: 'var(--text-primary)' }}>Forest Canopy</span>
                  <span className="data-mono" style={{ color: 'var(--text-secondary)' }}>12.1%</span>
                </div>
                <div style={{ height: '4px', backgroundColor: 'var(--border-strong)', borderRadius: '2px' }}>
                  <div style={{ width: '12.1%', height: '100%', backgroundColor: 'var(--border-subtle)' }}></div>
                </div>
              </div>

            </div>
          </div>
        </div>

        {/* Graph Re-ranking */}
        <div className="panel" style={{ flex: 1 }}>
          <div className="panel-header">
            <span>Graph Re-ranking Network</span>
            <Network size={14} color="var(--text-tertiary)" />
          </div>
          <div className="panel-content" style={{ padding: '24px', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            
            {/* Mock Network Graph */}
            <div style={{ position: 'relative', width: '100%', height: '140px' }}>
              {/* Edges */}
              <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
                <line x1="50%" y1="50%" x2="20%" y2="20%" stroke="var(--border-strong)" strokeWidth="2" />
                <line x1="50%" y1="50%" x2="80%" y2="30%" stroke="var(--border-strong)" strokeWidth="2" />
                <line x1="50%" y1="50%" x2="70%" y2="80%" stroke="var(--border-strong)" strokeWidth="2" />
                <line x1="50%" y1="50%" x2="30%" y2="70%" stroke="var(--border-strong)" strokeWidth="2" />
              </svg>
              {/* Nodes */}
              <div style={{ position: 'absolute', top: '20%', left: '20%', transform: 'translate(-50%, -50%)', width: '16px', height: '16px', borderRadius: '50%', backgroundColor: 'var(--text-tertiary)' }}></div>
              <div style={{ position: 'absolute', top: '30%', left: '80%', transform: 'translate(-50%, -50%)', width: '16px', height: '16px', borderRadius: '50%', backgroundColor: 'var(--text-tertiary)' }}></div>
              <div style={{ position: 'absolute', top: '80%', left: '70%', transform: 'translate(-50%, -50%)', width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'var(--text-secondary)' }}></div>
              <div style={{ position: 'absolute', top: '70%', left: '30%', transform: 'translate(-50%, -50%)', width: '16px', height: '16px', borderRadius: '50%', backgroundColor: 'var(--text-tertiary)' }}></div>
              
              {/* Central Node */}
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '32px', height: '32px', borderRadius: '50%', backgroundColor: 'var(--accent-saffron)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ width: '16px', height: '16px', borderRadius: '50%', backgroundColor: '#0a0a0b' }}></div>
              </div>
            </div>

            <div style={{ position: 'absolute', bottom: '16px', right: '16px', display: 'flex', gap: '8px', alignItems: 'center' }}>
              <Info size={12} color="var(--text-tertiary)" />
              <span style={{ fontSize: '0.625rem', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>k-reciprocal neighbors</span>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}
