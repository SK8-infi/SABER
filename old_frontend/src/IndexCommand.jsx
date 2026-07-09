import React from 'react';
import './index.css';
import { 
  Database,
  Server,
  HardDrive,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  Clock,
  Layers
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, CartesianGrid, AreaChart, Area } from 'recharts';

const throughputData = [
  { time: '00:00', rate: 120 }, { time: '00:05', rate: 145 }, { time: '00:10', rate: 160 },
  { time: '00:15', rate: 210 }, { time: '00:20', rate: 195 }, { time: '00:25', rate: 240 },
  { time: '00:30', rate: 235 }, { time: '00:35', rate: 280 }, { time: '00:40', rate: 275 },
];

const datasets = [
  { id: 1, name: 'BigEarthNet-MM', size: '590K', modalities: 'S1, S2', status: 'synced', health: '99.8%' },
  { id: 2, name: 'DSRSID', size: '4.2M', modalities: 'PAN, MS', status: 'synced', health: '99.9%' },
  { id: 3, name: 'CBRSIR_VS', size: '1.1M', modalities: 'PAN, MS', status: 'indexing', health: '84.2%' },
  { id: 4, name: 'EnMAP_Hyper', size: '12K', modalities: 'HSI', status: 'pending', health: '-' },
  { id: 5, name: 'SABER_Live_Stream', size: '8.4M', modalities: 'SAR, RGB', status: 'synced', health: '100%' },
];

export default function IndexCommand() {
  return (
    <div className="dashboard-grid" style={{ gridTemplateColumns: '1fr 2fr' }}>
      
      {/* LEFT PANEL: DATA SOURCES */}
      <div className="panel" style={{ height: 'calc(100vh - 180px)' }}>
        <div className="panel-header">
          <span>Data Sources & Collections</span>
          <Database size={14} color="var(--text-tertiary)" />
        </div>
        <div className="panel-content" style={{ padding: '0' }}>
          <table className="data-table" style={{ width: '100%' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-strong)', textAlign: 'left' }}>
                <th style={{ padding: '16px 24px', fontSize: '0.6875rem', color: 'var(--text-tertiary)' }}>COLLECTION</th>
                <th style={{ padding: '16px 24px', fontSize: '0.6875rem', color: 'var(--text-tertiary)' }}>SIZE</th>
                <th style={{ padding: '16px 24px', fontSize: '0.6875rem', color: 'var(--text-tertiary)' }}>MODALITIES</th>
                <th style={{ padding: '16px 24px', fontSize: '0.6875rem', color: 'var(--text-tertiary)' }}>STATUS</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map(ds => (
                <tr key={ds.id} style={{ borderBottom: '1px solid var(--border-subtle)', backgroundColor: ds.status === 'indexing' ? 'rgba(229, 141, 87, 0.05)' : 'transparent' }}>
                  <td style={{ padding: '16px 24px', color: 'var(--text-primary)', fontWeight: 500 }}>{ds.name}</td>
                  <td style={{ padding: '16px 24px', fontFamily: 'JetBrains Mono', color: 'var(--text-secondary)' }}>{ds.size}</td>
                  <td style={{ padding: '16px 24px', fontFamily: 'JetBrains Mono', color: 'var(--text-secondary)' }}>{ds.modalities}</td>
                  <td style={{ padding: '16px 24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {ds.status === 'synced' && <CheckCircle2 size={14} color="#a1a1aa" />}
                    {ds.status === 'indexing' && <RefreshCw size={14} color="var(--accent-saffron)" className="spin" />}
                    {ds.status === 'pending' && <Clock size={14} color="var(--text-tertiary)" />}
                    <span style={{ fontSize: '0.6875rem', textTransform: 'uppercase', color: ds.status === 'indexing' ? 'var(--accent-saffron)' : 'var(--text-secondary)' }}>
                      {ds.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ padding: '24px', marginTop: 'auto', borderTop: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.6875rem', color: 'var(--text-tertiary)' }}>
              <span>STORAGE FOOTPRINT</span>
              <span className="data-mono">84.2 TB / 100 TB</span>
            </div>
            <div style={{ height: '4px', backgroundColor: 'var(--border-strong)', borderRadius: '2px', overflow: 'hidden' }}>
              <div style={{ width: '84.2%', height: '100%', backgroundColor: 'var(--text-secondary)' }}></div>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT PANEL: TELEMETRY & HEALTH */}
      <div className="main-content">
        
        {/* Metric Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
          <div className="panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', fontWeight: 600 }}>TOTAL SCENES</span>
            <div style={{ display: 'flex', alignItems: 'end', gap: '12px' }}>
              <span style={{ fontSize: '2rem', fontFamily: 'JetBrains Mono', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>14.2M</span>
              <span style={{ fontSize: '0.8125rem', color: 'var(--accent-saffron)', marginBottom: '4px' }}>+12K/hr</span>
            </div>
          </div>
          <div className="panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', fontWeight: 600 }}>ACTIVE INDEX</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', height: '100%' }}>
              <Layers size={24} color="var(--text-secondary)" />
              <div>
                <div style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--text-primary)' }}>FAISS IVF-PQ</div>
                <div style={{ fontSize: '0.6875rem', fontFamily: 'JetBrains Mono', color: 'var(--text-secondary)' }}>FastScan / 4-bit</div>
              </div>
            </div>
          </div>
          <div className="panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)', fontWeight: 600 }}>RAM USAGE</span>
            <div style={{ display: 'flex', alignItems: 'end', gap: '12px' }}>
              <span style={{ fontSize: '2rem', fontFamily: 'JetBrains Mono', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>42.1</span>
              <span style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>GB</span>
            </div>
          </div>
        </div>

        {/* Throughput Chart */}
        <div className="panel" style={{ flex: 1 }}>
          <div className="panel-header">
            <span>Indexing Throughput (Scenes / Sec)</span>
            <Cpu size={14} color="var(--text-tertiary)" />
          </div>
          <div className="panel-content" style={{ padding: '24px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={throughputData}>
                <defs>
                  <linearGradient id="colorRate" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#e58d57" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#e58d57" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#71717a', fontFamily: 'JetBrains Mono' }} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#71717a', fontFamily: 'JetBrains Mono' }} dx={-10} />
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: '#0a0a0b', border: '1px solid #27272a', fontFamily: 'JetBrains Mono', fontSize: '12px' }}
                  itemStyle={{ color: '#e58d57' }}
                />
                <Area type="monotone" dataKey="rate" stroke="#e58d57" strokeWidth={2} fillOpacity={1} fill="url(#colorRate)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Health & Actions */}
        <div style={{ display: 'flex', gap: '24px' }}>
          <div className="panel" style={{ flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: '16px 24px' }}>
            <div style={{ display: 'flex', gap: '32px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8125rem' }}>
                <Server size={14} color="#a1a1aa" />
                <span style={{ color: 'var(--text-secondary)' }}>Backbone: <span style={{ color: 'var(--text-primary)' }}>DOFA ViT-B/16 (LoRA)</span></span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8125rem' }}>
                <HardDrive size={14} color="#a1a1aa" />
                <span style={{ color: 'var(--text-secondary)' }}>DB Connection: <span style={{ color: 'var(--text-primary)' }}>Stable</span></span>
              </div>
            </div>
            <button className="primary-btn" style={{ width: 'auto', padding: '12px 24px' }}>
              <RefreshCw size={14} /> REBUILD INDEX
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
