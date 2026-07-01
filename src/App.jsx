import React, { useState } from 'react';
import './index.css';
import RetrievalDashboard from './RetrievalDashboard';
import IndexCommand from './IndexCommand';
import SceneAnalytics from './SceneAnalytics';

function App() {
  const [activeView, setActiveView] = useState('retrieval');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', backgroundColor: 'var(--bg-color)' }}>
      
      {/* GLOBAL NAVIGATION SHELL */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 32px', borderBottom: '1px solid var(--border-subtle)', backgroundColor: 'var(--panel-bg)' }}>
        <div style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-primary)', fontWeight: '700', fontSize: '1.25rem', letterSpacing: '-0.02em', borderRight: '1px solid var(--border-strong)', paddingRight: '32px' }}>
            <img src="/logo.png" alt="SABER Logo" style={{ height: '24px', width: '24px', objectFit: 'contain' }} />
            <span>SABER</span>
          </div>
          
          <div style={{ display: 'flex', gap: '24px' }}>
            <button 
              onClick={() => setActiveView('retrieval')}
              style={{ 
                background: 'none', border: 'none', color: activeView === 'retrieval' ? 'var(--accent-saffron)' : 'var(--text-secondary)', 
                fontSize: '0.8125rem', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase', cursor: 'pointer',
                borderBottom: activeView === 'retrieval' ? '2px solid var(--accent-saffron)' : '2px solid transparent',
                paddingBottom: '4px', transition: 'all 0.2s'
              }}
            >
              Precision Retrieval
            </button>
            <button 
              onClick={() => setActiveView('analytics')}
              style={{ 
                background: 'none', border: 'none', color: activeView === 'analytics' ? 'var(--accent-saffron)' : 'var(--text-secondary)', 
                fontSize: '0.8125rem', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase', cursor: 'pointer',
                borderBottom: activeView === 'analytics' ? '2px solid var(--accent-saffron)' : '2px solid transparent',
                paddingBottom: '4px', transition: 'all 0.2s'
              }}
            >
              Scene Analytics
            </button>
            <button 
              onClick={() => setActiveView('index')}
              style={{ 
                background: 'none', border: 'none', color: activeView === 'index' ? 'var(--accent-saffron)' : 'var(--text-secondary)', 
                fontSize: '0.8125rem', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase', cursor: 'pointer',
                borderBottom: activeView === 'index' ? '2px solid var(--accent-saffron)' : '2px solid transparent',
                paddingBottom: '4px', transition: 'all 0.2s'
              }}
            >
              Index Command
            </button>
          </div>
        </div>

        <div style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--text-secondary)' }}></span>
          OFFLINE
        </div>
      </div>

      {/* ACTIVE VIEW */}
      <div style={{ flex: 1 }}>
        {activeView === 'retrieval' && <RetrievalDashboard />}
        {activeView === 'index' && <div className="dashboard-container"><IndexCommand /></div>}
        {activeView === 'analytics' && <div className="dashboard-container"><SceneAnalytics /></div>}
      </div>

    </div>
  );
}

export default App;
