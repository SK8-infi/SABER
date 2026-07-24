import React, { useState, useEffect } from 'react';
import { Database, Filter, Layers, Info } from 'lucide-react';

export default function DatasetExplorer() {
  const [activeDataset, setActiveDataset] = useState('ben14k');
  const [stats, setStats] = useState(null);
  const [selectedClass, setSelectedClass] = useState(null);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`/api/dataset/stats?name=${activeDataset}`)
      .then((res) => res.json())
      .then((data) => setStats(data))
      .catch((err) => console.error("Error loading dataset stats:", err));

    loadSamples();
  }, [activeDataset, selectedClass]);

  const loadSamples = async () => {
    setLoading(true);
    try {
      let url = `/api/dataset/samples?dataset_name=${activeDataset}&limit=12`;
      if (selectedClass !== null) {
        url += `&class_index=${selectedClass}`;
      }
      const res = await fetch(url);
      const data = await res.json();
      setSamples(data.items || []);
    } catch (err) {
      console.error("Error loading dataset samples:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      
      {/* Header & Dataset Toggle */}
      <div className="scientific-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Database className="card-title-icon" size={16} />
            <span style={{ fontSize: '1.1rem', fontWeight: 700 }}>Multi-Sensor Dataset & Taxonomy Explorer</span>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Inspect dataset sensors, spectral band configurations, CORINE 19 multi-label taxonomy, and sample image rasters.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            className={`secondary-btn ${activeDataset === 'ben14k' ? 'active' : ''}`} 
            style={{ backgroundColor: activeDataset === 'ben14k' ? 'var(--accent-saffron-glow)' : 'transparent', borderColor: activeDataset === 'ben14k' ? 'var(--accent-saffron)' : 'var(--border-color)' }}
            onClick={() => { setActiveDataset('ben14k'); setSelectedClass(null); }}
          >
            BEN-14K (Sentinel-1/2)
          </button>
          <button 
            className={`secondary-btn ${activeDataset === 'dsrsid' ? 'active' : ''}`}
            style={{ backgroundColor: activeDataset === 'dsrsid' ? 'var(--accent-cyan-glow)' : 'transparent', borderColor: activeDataset === 'dsrsid' ? 'var(--accent-cyan)' : 'var(--border-color)' }}
            onClick={() => { setActiveDataset('dsrsid'); setSelectedClass(null); }}
          >
            DSRSID (Gaofen-1)
          </button>
        </div>
      </div>

      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '20px' }}>
          
          {/* LEFT: Taxonomy & Class Filter */}
          <div className="scientific-card">
            <div className="card-header">
              <span className="card-title" style={{ fontSize: '0.85rem' }}>
                <Filter size={14} /> Semantic Class Taxonomy ({stats.num_classes} Classes)
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '480px', overflowY: 'auto' }}>
              <button 
                className="secondary-btn" 
                style={{ justifyContent: 'flex-start', fontSize: '0.75rem', borderColor: selectedClass === null ? 'var(--accent-saffron)' : 'var(--border-color)' }}
                onClick={() => setSelectedClass(null)}
              >
                ALL CLASSES ({stats.total_samples} Scenes)
              </button>

              {stats.classes.map((clsName, idx) => (
                <button 
                  key={idx}
                  className="secondary-btn"
                  style={{ justifyContent: 'flex-start', fontSize: '0.73rem', borderColor: selectedClass === idx ? 'var(--accent-cyan)' : 'var(--border-color)', color: selectedClass === idx ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}
                  onClick={() => setSelectedClass(idx)}
                >
                  <span className="data-mono" style={{ color: 'var(--text-muted)', marginRight: '6px' }}>#{idx + 1}</span> {clsName}
                </button>
              ))}
            </div>
          </div>

          {/* RIGHT: Sample Gallery Browser */}
          <div className="scientific-card">
            <div className="card-header">
              <span className="card-title" style={{ fontSize: '0.85rem' }}>
                Dataset Sample Browser {selectedClass !== null && `(Class: ${stats.classes[selectedClass]})`}
              </span>
              <span className="mono-tag cyan">SHOWING {samples.length} SAMPLES</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px' }}>
              {samples.map((item) => (
                <div key={item.index} style={{ backgroundColor: 'var(--bg-dark)', borderRadius: '6px', border: '1px solid var(--border-color)', overflow: 'hidden' }}>
                  <div style={{ width: '100%', height: '110px', overflow: 'hidden' }}>
                    <img src={item.thumbnail} alt={item.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  </div>
                  <div style={{ padding: '8px' }}>
                    <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {item.name}
                    </div>
                    <div className="class-chips" style={{ marginTop: '4px' }}>
                      {item.active_classes.slice(0, 2).map((c, i) => (
                        <span key={i} className="class-chip" style={{ fontSize: '0.6rem' }}>{c}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}

    </div>
  );
}
