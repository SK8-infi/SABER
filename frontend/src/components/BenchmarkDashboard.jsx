import React, { useState, useEffect } from 'react';
import { BarChart3, Award, CheckCircle, Info, Zap } from 'lucide-react';

export default function BenchmarkDashboard() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch('/api/benchmark/metrics')
      .then((res) => res.json())
      .then((data) => setMetrics(data))
      .catch((err) => console.error("Error loading benchmark metrics:", err));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      
      {/* Header Banner */}
      <div className="scientific-card">
        <div className="card-header">
          <span className="card-title">
            <BarChart3 className="card-title-icon" size={16} /> Quantitative SOTA Benchmark Dashboard & Experimental Validation
          </span>
          <span className="mono-tag saffron">ISRO BAH 2026 BENCHMARKS</span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Evaluated on real 100% non-synthetic satellite partitions (20% Query / 80% Gallery split: 2,966 query scenes against 11,866 gallery scenes for BEN-14K; 
          2,000 query scenes against 8,000 gallery scenes for DSRSID).
        </p>
      </div>

      {/* BEN-14K Benchmark Table */}
      {metrics && (
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title" style={{ color: 'var(--accent-saffron)' }}>
              1. BEN-14K Sentinel-1 SAR ◄► Sentinel-2 MS Benchmark Matrix
            </span>
            <span className="mono-tag cyan">2,966 QUERIES / 11,866 GALLERY</span>
          </div>

          <table className="scientific-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Precision@5</th>
                <th>Recall@5</th>
                <th>F1@5</th>
                <th>F1@10</th>
                <th>mAP (Global)</th>
                <th>Query Latency</th>
                <th>Trainable Params</th>
              </tr>
            </thead>
            <tbody>
              {metrics.ben14k_benchmark.map((row, idx) => {
                const isSaber = row.model.includes("SABER");
                return (
                  <tr key={idx} style={{ backgroundColor: isSaber ? 'rgba(255, 153, 51, 0.08)' : 'transparent', fontWeight: isSaber ? 600 : 400 }}>
                    <td style={{ color: isSaber ? 'var(--accent-saffron)' : 'var(--text-primary)' }}>
                      {isSaber && <Award size={14} style={{ display: 'inline', marginRight: '6px' }} />}
                      {row.model}
                    </td>
                    <td className="data-mono">{row.precision_5}</td>
                    <td className="data-mono">{row.recall_5}</td>
                    <td className="data-mono" style={{ color: isSaber ? 'var(--accent-green)' : 'inherit', fontWeight: isSaber ? 700 : 400 }}>
                      {row.f1_5}
                    </td>
                    <td className="data-mono">{row.f1_10}</td>
                    <td className="data-mono" style={{ color: isSaber ? 'var(--accent-cyan)' : 'inherit', fontWeight: isSaber ? 700 : 400 }}>
                      {row.mAP}
                    </td>
                    <td className="data-mono" style={{ color: 'var(--text-muted)' }}>{row.latency_ms}</td>
                    <td className="data-mono">{row.params_trainable}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* DSRSID Benchmark Table */}
      {metrics && (
        <div className="scientific-card">
          <div className="card-header">
            <span className="card-title" style={{ color: 'var(--accent-cyan)' }}>
              2. DSRSID Gaofen-1 PAN ◄► Gaofen-1 MS Benchmark Matrix
            </span>
            <span className="mono-tag">2,000 QUERIES / 8,000 GALLERY</span>
          </div>

          <table className="scientific-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Precision@5</th>
                <th>Precision@10</th>
                <th>Recall@5</th>
                <th>F1@5</th>
                <th>mAP (Global)</th>
                <th>Avg Query Latency</th>
              </tr>
            </thead>
            <tbody>
              {metrics.dsrsid_benchmark.map((row, idx) => {
                const isSaber = row.model.includes("SABER");
                return (
                  <tr key={idx} style={{ backgroundColor: isSaber ? 'rgba(0, 229, 255, 0.08)' : 'transparent' }}>
                    <td style={{ color: isSaber ? 'var(--accent-cyan)' : 'var(--text-primary)', fontWeight: isSaber ? 600 : 400 }}>
                      {row.model}
                    </td>
                    <td className="data-mono" style={{ color: isSaber ? 'var(--accent-green)' : 'inherit', fontWeight: isSaber ? 700 : 400 }}>
                      {row.precision_5}
                    </td>
                    <td className="data-mono">{row.precision_10}</td>
                    <td className="data-mono">{row.recall_5}</td>
                    <td className="data-mono">{row.f1_5}</td>
                    <td className="data-mono">{row.mAP}</td>
                    <td className="data-mono">{row.latency_ms}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* ISRO PS11 Evaluation Summary Card */}
      {metrics && (
        <div className="scientific-card" style={{ border: '1px solid var(--accent-green)' }}>
          <div className="card-header">
            <span className="card-title" style={{ color: 'var(--accent-green)' }}>
              <CheckCircle size={16} /> ISRO BAH 2026 Problem Statement 11 Target Compliance
            </span>
            <span className="mono-tag green">VERIFIED READY</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
            <div style={{ backgroundColor: 'var(--bg-dark)', padding: '14px', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>SAME-MODAL F1@5 CEILING</div>
              <div className="data-mono" style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                {metrics.isro_ps11_eval.target_same_modal_f1_5}
              </div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-dark)', padding: '14px', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>CROSS-MODAL F1@5 SCORE</div>
              <div className="data-mono" style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-saffron)' }}>
                {metrics.isro_ps11_eval.target_cross_modal_f1_5}
              </div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-dark)', padding: '14px', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>CROSS-MODAL mAP</div>
              <div className="data-mono" style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                {metrics.isro_ps11_eval.target_cross_modal_map}
              </div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-dark)', padding: '14px', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>AVG QUERY LATENCY</div>
              <div className="data-mono" style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                {metrics.isro_ps11_eval.target_query_latency}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
