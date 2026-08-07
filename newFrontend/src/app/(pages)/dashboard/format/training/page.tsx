'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import {
  Activity,
  Zap,
  TrendingDown,
  Award,
  Cpu,
  CheckCircle2,
  BarChart3,
  Sliders,
  Database,
} from 'lucide-react'

import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

/* ── 20-Epoch Main Encoder Training Data ─────────────────────── */
const encoderEpochData = [
  { epoch: 1,  loss: 27.3668, jaccard: 0.5047, rank: 2.4504, trip: 0.2988, sigr: 0.4570, clas: 0.8103, inva: 0.1894, vari: 0.7332, cova: 0.2527, lr: 0.000340 },
  { epoch: 2,  loss: 25.9101, jaccard: 0.4166, rank: 2.3337, trip: 0.2988, sigr: 0.4581, clas: 0.3727, inva: 0.1363, vari: 0.7438, cova: 0.1852, lr: 0.000670 },
  { epoch: 3,  loss: 25.5505, jaccard: 0.3995, rank: 2.2988, trip: 0.2982, sigr: 0.4593, clas: 0.2270, inva: 0.1280, vari: 0.7422, cova: 0.2037, lr: 0.001000 },
  { epoch: 4,  loss: 25.4731, jaccard: 0.4026, rank: 2.2896, trip: 0.2979, sigr: 0.4608, clas: 0.2027, inva: 0.1239, vari: 0.7419, cova: 0.2156, lr: 0.000991 },
  { epoch: 5,  loss: 25.2843, jaccard: 0.3986, rank: 2.2456, trip: 0.2974, sigr: 0.4557, clas: 0.1921, inva: 0.1212, vari: 0.7381, cova: 0.2304, lr: 0.000966 },
  { epoch: 6,  loss: 25.0882, jaccard: 0.3934, rank: 2.2249, trip: 0.2970, sigr: 0.4571, clas: 0.1855, inva: 0.1196, vari: 0.7319, cova: 0.2463, lr: 0.000925 },
  { epoch: 7,  loss: 24.8329, jaccard: 0.3826, rank: 2.1828, trip: 0.2965, sigr: 0.4507, clas: 0.1807, inva: 0.1201, vari: 0.7222, cova: 0.2815, lr: 0.000870 },
  { epoch: 8,  loss: 24.6082, jaccard: 0.3746, rank: 2.1489, trip: 0.2958, sigr: 0.4451, clas: 0.1772, inva: 0.1190, vari: 0.7140, cova: 0.3151, lr: 0.000802 },
  { epoch: 9,  loss: 24.4160, jaccard: 0.3684, rank: 2.1134, trip: 0.2954, sigr: 0.4393, clas: 0.1736, inva: 0.1181, vari: 0.7068, cova: 0.3509, lr: 0.000723 },
  { epoch: 10, loss: 24.2498, jaccard: 0.3588, rank: 2.0897, trip: 0.2946, sigr: 0.4343, clas: 0.1704, inva: 0.1174, vari: 0.6992, cova: 0.3975, lr: 0.000637 },
  { epoch: 11, loss: 24.1150, jaccard: 0.3536, rank: 2.0631, trip: 0.2942, sigr: 0.4313, clas: 0.1675, inva: 0.1152, vari: 0.6940, cova: 0.4386, lr: 0.000547 },
  { epoch: 12, loss: 24.0276, jaccard: 0.3489, rank: 2.0457, trip: 0.2938, sigr: 0.4297, clas: 0.1658, inva: 0.1152, vari: 0.6882, cova: 0.4865, lr: 0.000454 },
  { epoch: 13, loss: 23.9065, jaccard: 0.3410, rank: 2.0082, trip: 0.2935, sigr: 0.4249, clas: 0.1642, inva: 0.1136, vari: 0.6848, cova: 0.5173, lr: 0.000364 },
  { epoch: 14, loss: 23.8307, jaccard: 0.3358, rank: 1.9899, trip: 0.2929, sigr: 0.4242, clas: 0.1622, inva: 0.1126, vari: 0.6821, cova: 0.5407, lr: 0.000278 },
  { epoch: 15, loss: 23.7607, jaccard: 0.3322, rank: 1.9633, trip: 0.2930, sigr: 0.4248, clas: 0.1605, inva: 0.1113, vari: 0.6806, cova: 0.5590, lr: 0.000199 },
  { epoch: 16, loss: 23.7202, jaccard: 0.3279, rank: 1.9631, trip: 0.2927, sigr: 0.4237, clas: 0.1582, inva: 0.1106, vari: 0.6787, cova: 0.5739, lr: 0.000131 },
  { epoch: 17, loss: 23.6551, jaccard: 0.3246, rank: 1.9390, trip: 0.2924, sigr: 0.4188, clas: 0.1570, inva: 0.1100, vari: 0.6775, cova: 0.5821, lr: 0.000076 },
  { epoch: 18, loss: 23.6309, jaccard: 0.3213, rank: 1.9343, trip: 0.2922, sigr: 0.4218, clas: 0.1563, inva: 0.1096, vari: 0.6765, cova: 0.5931, lr: 0.000035 },
  { epoch: 19, loss: 23.6242, jaccard: 0.3212, rank: 1.9287, trip: 0.2922, sigr: 0.4188, clas: 0.1560, inva: 0.1090, vari: 0.6760, cova: 0.6043, lr: 0.000010 },
  { epoch: 20, loss: 23.5892, jaccard: 0.3200, rank: 1.9087, trip: 0.2921, sigr: 0.4214, clas: 0.1552, inva: 0.1095, vari: 0.6754, cova: 0.6073, lr: 0.000001 },
]

/* ── 80-Epoch CFM Bridge Training Subsample Data ───────────────── */
const bridgeData = [
  { epoch: 1,  loss: 1.2497,  f1_step1: 73.19, f1_step10: 73.33 },
  { epoch: 2,  loss: -0.0965, f1_step1: 74.77, f1_step10: 74.82 },
  { epoch: 3,  loss: -0.4514, f1_step1: 75.20, f1_step10: 75.25 },
  { epoch: 4,  loss: -0.5865, f1_step1: 75.28, f1_step10: 75.32 },
  { epoch: 5,  loss: -0.6811, f1_step1: 75.27, f1_step10: 75.32 },
  { epoch: 10, loss: -0.9547, f1_step1: 75.06, f1_step10: 74.79 },
  { epoch: 15, loss: -1.0926, f1_step1: 74.89, f1_step10: 74.71 },
  { epoch: 20, loss: -1.1856, f1_step1: 75.05, f1_step10: 74.69 },
  { epoch: 25, loss: -1.2550, f1_step1: 74.92, f1_step10: 74.39 },
  { epoch: 30, loss: -1.3109, f1_step1: 75.11, f1_step10: 74.58 },
  { epoch: 35, loss: -1.3603, f1_step1: 74.95, f1_step10: 74.54 },
  { epoch: 40, loss: -1.4030, f1_step1: 75.40, f1_step10: 75.17 },
  { epoch: 45, loss: -1.4349, f1_step1: 75.08, f1_step10: 74.69 },
  { epoch: 50, loss: -1.4640, f1_step1: 75.08, f1_step10: 74.48 },
  { epoch: 55, loss: -1.4897, f1_step1: 75.12, f1_step10: 74.73 },
  { epoch: 60, loss: -1.5104, f1_step1: 75.42, f1_step10: 75.24 },
  { epoch: 65, loss: -1.5287, f1_step1: 75.27, f1_step10: 75.02 },
  { epoch: 70, loss: -1.5404, f1_step1: 75.43, f1_step10: 75.26 },
  { epoch: 75, loss: -1.5486, f1_step1: 75.26, f1_step10: 75.16 },
  { epoch: 80, loss: -1.5536, f1_step1: 75.19, f1_step10: 75.04 },
]

/* ── Benchmark Comparison Data ────────────────────────────────── */
/* ── Benchmark Comparison Data ────────────────────────────────── */
const benchmarkData = [
  { metric: 'S1 → S2 F1@5',   crossModal: 73.51, sameModal: 75.40 },
  { metric: 'S2 → S1 F1@5',   crossModal: 73.10, sameModal: 76.38 },
  { metric: 'Cross-Modal mAP', crossModal: 91.49, sameModal: 91.51 },
  { metric: 'Retrieval Latency', crossModal: 28.48, sameModal: 14.80 },
]

const peerComparisonTable = [
  { model: 'MAE', baseline: 'Standard Self-Supervised', s1s2: '~46.2%', s2s1: '~47.1%', s1s1: '~63.8%', s2s2: '~71.2%', map: '~58.3%', params: '100% (86M)' },
  { model: 'SatMAE', baseline: 'Satellite Masked Autoencoder', s1s2: '~51.4%', s2s1: '~52.3%', s1s1: '~68.1%', s2s2: '~75.4%', map: '~63.8%', params: '100% (86M)' },
  { model: 'MAE-RVSA', baseline: 'Rotated Variational Attention', s1s2: '~54.8%', s2s1: '~55.2%', s1s1: '~70.2%', s2s2: '~77.8%', map: '~66.1%', params: '100% (86M)' },
  { model: 'RemoteCLIP', baseline: 'IEEE TGRS 2024 (Contrastive)', s1s2: '49.80%', s2s1: '50.10%', s1s1: '—', s2s2: '—', map: '67.40%', params: '100% (149M)' },
  { model: 'X-JEPA', baseline: 'CVPR 2024 (Cross-Modal JEPA)', s1s2: '61.23%', s2s1: '63.73%', s1s1: '72.98%', s2s2: '82.65%', map: '71.95%', params: '100% (86M)' },
  { model: 'CR-JEPA', baseline: 'arXiv:2606.00706', s1s2: '75.82%', s2s1: '75.40%', s1s1: '75.11%', s2s2: '82.87%', map: '~78.50%', params: '100% (86M)' },
  { model: 'SABER (Ours)', baseline: 'DOFA + LoRA + CFM ODE Bridge', s1s2: '73.51%', s2s1: '73.10%', s1s1: '75.40%', s2s2: '76.38%', map: '91.49%', params: '0.26% (294.9K)' },
]

export default function TrainingDashboardPage() {
  const [activeTab, setActiveTab] = useState<'encoder' | 'bridge' | 'benchmark'>('encoder')

  return (
    <div className='w-full space-y-6'>
      {/* ── Top Header Banner ── */}
      <Card className='border-border/60 shadow-sm overflow-hidden border-t-4 border-t-[#FBBA72] bg-card/60 backdrop-blur-xs'>
        <CardContent className='p-6 space-y-4'>
          <div className='flex flex-wrap items-center justify-between gap-4'>
            <div className='space-y-1.5'>
              <div className='flex items-center gap-2.5 flex-wrap'>
                <h1 className='text-xl font-bold tracking-tight text-foreground font-sans'>
                  SABER Latest SOTA Benchmark & Training Telemetry
                </h1>
                <Badge className='bg-[#FBBA72]/15 text-[#FBBA72] border-[#FBBA72]/40 font-semibold px-2.5 py-0.5 text-xs rounded-full font-sans'>
                  Official BEN-14K Real Data Partition
                </Badge>
              </div>
              <p className='text-xs text-muted-foreground font-sans max-w-2xl'>
                Complete convergence telemetry for 20-Epoch Multi-Modal Encoder + 80-Epoch Continuous Flow Matching (CFM) Latent Bridge trained on BEN-14K.
              </p>
            </div>

            {/* Quick Metrics Pill */}
            <div className='flex items-center gap-2 bg-muted/30 border border-border/40 p-2 rounded-xl text-xs font-sans'>
              <div className='flex items-center gap-1.5 px-2.5 py-1 bg-[#FBBA72]/10 rounded-lg border border-[#FBBA72]/30'>
                <Award className='size-3.5 text-[#FBBA72]' />
                <span className='font-bold text-[#FBBA72]'>73.51% F1@5</span>
              </div>
              <div className='flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/10 rounded-lg border border-emerald-500/30'>
                <Zap className='size-3.5 text-emerald-400' />
                <span className='font-bold text-emerald-400'>91.49% mAP</span>
              </div>
            </div>
          </div>

          {/* Quick Specs Grid */}
          <div className='grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2'>
            <div className='flex flex-col gap-0.5 rounded-lg bg-muted/20 border border-border/30 p-2.5'>
              <span className='text-[10px] font-semibold uppercase tracking-wider text-muted-foreground font-sans flex items-center gap-1'>
                <Cpu className='size-3 text-[#FBBA72]' /> Model Backbone
              </span>
              <span className='text-xs font-bold text-foreground font-sans'>DOFA ViT-Base/16</span>
            </div>
            <div className='flex flex-col gap-0.5 rounded-lg bg-muted/20 border border-border/30 p-2.5'>
              <span className='text-[10px] font-semibold uppercase tracking-wider text-muted-foreground font-sans flex items-center gap-1'>
                <Sliders className='size-3 text-[#FBBA72]' /> Trainable Params
              </span>
              <span className='text-xs font-bold text-foreground font-sans'>294.9K / 111.6M (0.26% LoRA)</span>
            </div>
            <div className='flex flex-col gap-0.5 rounded-lg bg-muted/20 border border-border/30 p-2.5'>
              <span className='text-[10px] font-semibold uppercase tracking-wider text-muted-foreground font-sans flex items-center gap-1'>
                <Database className='size-3 text-[#FBBA72]' /> Dataset Size
              </span>
              <span className='text-xs font-bold text-foreground font-sans'>BEN-14K (14,832 Samples)</span>
            </div>
            <div className='flex flex-col gap-0.5 rounded-lg bg-muted/20 border border-border/30 p-2.5'>
              <span className='text-[10px] font-semibold uppercase tracking-wider text-muted-foreground font-sans flex items-center gap-1'>
                <Activity className='size-3 text-[#FBBA72]' /> Total Latency
              </span>
              <span className='text-xs font-bold text-foreground font-sans'>28.48 ms (0.97ms FAISS)</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── KPI Stat Cards ── */}
      <div className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4'>
        <Card className='border-border/60 hover:border-[#FBBA72]/40 transition-colors'>
          <CardContent className='p-4 space-y-2'>
            <div className='flex items-center justify-between'>
              <span className='text-xs font-semibold uppercase tracking-wider text-muted-foreground font-sans'>Encoder Loss</span>
              <TrendingDown className='size-4 text-emerald-400' />
            </div>
            <div className='flex items-baseline gap-2'>
              <span className='text-2xl font-bold text-foreground font-sans'>23.589</span>
              <span className='text-xs font-medium text-emerald-400 font-sans'>-13.8% from 27.37</span>
            </div>
            <p className='text-[10px] text-muted-foreground font-sans'>Multi-task supervised loss convergence across 20 epochs</p>
          </CardContent>
        </Card>

        <Card className='border-border/60 hover:border-[#FBBA72]/40 transition-colors'>
          <CardContent className='p-4 space-y-2'>
            <div className='flex items-center justify-between'>
              <span className='text-xs font-semibold uppercase tracking-wider text-muted-foreground font-sans'>Cross-Modal F1@5</span>
              <Award className='size-4 text-[#FBBA72]' />
            </div>
            <div className='flex items-baseline gap-2'>
              <span className='text-2xl font-bold text-[#FBBA72] font-sans'>73.51%</span>
              <span className='text-xs font-medium text-[#FBBA72] font-sans'>Sentinel-1 SAR → Sentinel-2</span>
            </div>
            <p className='text-[10px] text-muted-foreground font-sans'>Cross-modal retrieval precision on held-out test split</p>
          </CardContent>
        </Card>

        <Card className='border-border/60 hover:border-[#FBBA72]/40 transition-colors'>
          <CardContent className='p-4 space-y-2'>
            <div className='flex items-center justify-between'>
              <span className='text-xs font-semibold uppercase tracking-wider text-muted-foreground font-sans'>Mean Avg Precision</span>
              <CheckCircle2 className='size-4 text-emerald-400' />
            </div>
            <div className='flex items-baseline gap-2'>
              <span className='text-2xl font-bold text-emerald-400 font-sans'>91.49%</span>
              <span className='text-xs font-medium text-muted-foreground font-sans'>91.51% Raw Alignment</span>
            </div>
            <p className='text-[10px] text-muted-foreground font-sans'>Near-perfect semantic rank accuracy across held-out test split</p>
          </CardContent>
        </Card>

        <Card className='border-border/60 hover:border-[#FBBA72]/40 transition-colors'>
          <CardContent className='p-4 space-y-2'>
            <div className='flex items-center justify-between'>
              <span className='text-xs font-semibold uppercase tracking-wider text-muted-foreground font-sans'>Query Latency</span>
              <Zap className='size-4 text-[#FBBA72]' />
            </div>
            <div className='flex items-baseline gap-2'>
              <span className='text-2xl font-bold text-[#FBBA72] font-sans'>28.48 ms</span>
              <span className='text-xs font-medium text-muted-foreground font-sans'>0.97ms FAISS</span>
            </div>
            <p className='text-[10px] text-muted-foreground font-sans'>End-to-end GPU ODE translation and FAISS vector retrieval</p>
          </CardContent>
        </Card>
      </div>

      {/* ── Navigation Tab Switcher ── */}
      <div className='flex items-center gap-2 border-b border-border/40 pb-2'>
        <button
          onClick={() => setActiveTab('encoder')}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-colors font-sans cursor-pointer ${
            activeTab === 'encoder'
              ? 'bg-[#FBBA72]/15 text-[#FBBA72] border border-[#FBBA72]/30'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
          }`}
        >
          <Activity className='size-3.5' />
          <span>Main Encoder Training (20 Epochs)</span>
        </button>
        <button
          onClick={() => setActiveTab('bridge')}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-colors font-sans cursor-pointer ${
            activeTab === 'bridge'
              ? 'bg-[#FBBA72]/15 text-[#FBBA72] border border-[#FBBA72]/30'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
          }`}
        >
          <Zap className='size-3.5' />
          <span>CFM Latent Bridge (80 Epochs)</span>
        </button>
        <button
          onClick={() => setActiveTab('benchmark')}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-colors font-sans cursor-pointer ${
            activeTab === 'benchmark'
              ? 'bg-[#FBBA72]/15 text-[#FBBA72] border border-[#FBBA72]/30'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
          }`}
        >
          <BarChart3 className='size-3.5' />
          <span>Benchmark Evaluation Metrics</span>
        </button>
      </div>

      {/* ── TAB 1: Main Encoder Training ── */}
      {activeTab === 'encoder' && (
        <div className='space-y-6'>
          {/* Main Total Loss & LR Chart */}
          <Card className='border-border/60'>
            <CardContent className='p-6 space-y-4'>
              <div className='flex items-center justify-between flex-wrap gap-2'>
                <div>
                  <h2 className='text-base font-bold text-foreground font-sans'>Total Training Loss & Cosine Learning Rate</h2>
                  <p className='text-xs text-muted-foreground font-sans'>Monotonic convergence across 20 training epochs with warm-up cosine decay</p>
                </div>
                <div className='flex items-center gap-3 text-xs font-sans'>
                  <span className='flex items-center gap-1 text-[#FBBA72] font-semibold'>
                    <span className='size-2 rounded-full bg-[#FBBA72]' /> Total Loss
                  </span>
                  <span className='flex items-center gap-1 text-sky-400 font-semibold'>
                    <span className='size-2 rounded-full bg-sky-400' /> Learning Rate (x1e-3)
                  </span>
                </div>
              </div>

              <div className='h-72 w-full'>
                <ResponsiveContainer width='100%' height='100%'>
                  <AreaChart data={encoderEpochData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id='lossGrad' x1='0' y1='0' x2='0' y2='1'>
                        <stop offset='5%' stopColor='#FBBA72' stopOpacity={0.4} />
                        <stop offset='95%' stopColor='#FBBA72' stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray='3 3' stroke='currentColor' opacity={0.1} />
                    <XAxis dataKey='epoch' tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <YAxis yAxisId='left' domain={[22, 28]} tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <YAxis yAxisId='right' orientation='right' domain={[0, 0.0012]} tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'oklch(0.18 0.004 285.823)', borderColor: 'oklch(0.27 0.005 286.033)', borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Area yAxisId='left' type='monotone' dataKey='loss' name='Total Loss' stroke='#FBBA72' strokeWidth={2.5} fillOpacity={1} fill='url(#lossGrad)' />
                    <Line yAxisId='right' type='monotone' dataKey='lr' name='Learning Rate' stroke='#38bdf8' strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Multi-Component Loss Breakdown */}
          <Card className='border-border/60'>
            <CardContent className='p-6 space-y-4'>
              <div>
                <h2 className='text-base font-bold text-foreground font-sans'>Multi-Task Loss Component Progression</h2>
                <p className='text-xs text-muted-foreground font-sans'>Disaggregated breakdown of Jaccard Target Loss, Ranking KL-Divergence, Land Cover Classification, and VICReg Regularization terms</p>
              </div>

              <div className='h-80 w-full'>
                <ResponsiveContainer width='100%' height='100%'>
                  <LineChart data={encoderEpochData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray='3 3' stroke='currentColor' opacity={0.1} />
                    <XAxis dataKey='epoch' tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <YAxis domain={[0, 2.5]} tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <Tooltip contentStyle={{ backgroundColor: 'oklch(0.18 0.004 285.823)', borderColor: 'oklch(0.27 0.005 286.033)', borderRadius: '8px', fontSize: '12px' }} />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Line type='monotone' dataKey='rank' name='Neighborhood Rank' stroke='#f43f5e' strokeWidth={2} dot={false} />
                    <Line type='monotone' dataKey='clas' name='Classification' stroke='#a855f7' strokeWidth={2} dot={false} />
                    <Line type='monotone' dataKey='jaccard' name='Jaccard Target' stroke='#FBBA72' strokeWidth={2} dot={false} />
                    <Line type='monotone' dataKey='vari' name='VICReg Variance' stroke='#34d399' strokeWidth={1.5} strokeDasharray='3 3' dot={false} />
                    <Line type='monotone' dataKey='cova' name='VICReg Covariance' stroke='#38bdf8' strokeWidth={1.5} strokeDasharray='3 3' dot={false} />
                    <Line type='monotone' dataKey='inva' name='VICReg Invariance' stroke='#f59e0b' strokeWidth={1.5} strokeDasharray='3 3' dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ── TAB 2: CFM Latent Bridge ── */}
      {activeTab === 'bridge' && (
        <div className='space-y-6'>
          <Card className='border-border/60'>
            <CardContent className='p-6 space-y-4'>
              <div>
                <h2 className='text-base font-bold text-foreground font-sans'>CFM Latent Bridge Loss & F1@5 Retrieval Accuracy (80 Epochs)</h2>
                <p className='text-xs text-muted-foreground font-sans'>Continuous Flow Matching vector field optimization from Sentinel-1 SAR manifold to Sentinel-2 Optical space</p>
              </div>

              <div className='h-80 w-full'>
                <ResponsiveContainer width='100%' height='100%'>
                  <LineChart data={bridgeData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray='3 3' stroke='currentColor' opacity={0.1} />
                    <XAxis dataKey='epoch' tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <YAxis yAxisId='left' domain={[-1.7, 1.5]} tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <YAxis yAxisId='right' orientation='right' domain={[72, 77]} tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <Tooltip contentStyle={{ backgroundColor: 'oklch(0.18 0.004 285.823)', borderColor: 'oklch(0.27 0.005 286.033)', borderRadius: '8px', fontSize: '12px' }} />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Line yAxisId='left' type='monotone' dataKey='loss' name='CFM Vector Loss' stroke='#FBBA72' strokeWidth={2.5} />
                    <Line yAxisId='right' type='monotone' dataKey='f1_step1' name='1-Step Euler F1@5 (%)' stroke='#34d399' strokeWidth={2} />
                    <Line yAxisId='right' type='monotone' dataKey='f1_step10' name='10-Step ODE F1@5 (%)' stroke='#38bdf8' strokeWidth={2} strokeDasharray='4 4' />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ── TAB 3: Benchmark Evaluation Metrics ── */}
      {activeTab === 'benchmark' && (
        <div className='space-y-6'>
          <Card className='border-border/60'>
            <CardContent className='p-6 space-y-4'>
              <div>
                <h2 className='text-base font-bold text-foreground font-sans'>Cross-Modal vs Same-Modal Evaluation Benchmark</h2>
                <p className='text-xs text-muted-foreground font-sans'>Held-out test set comparison across top-5 and top-10 retrieval metrics</p>
              </div>

              <div className='h-80 w-full'>
                <ResponsiveContainer width='100%' height='100%'>
                  <BarChart data={benchmarkData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray='3 3' stroke='currentColor' opacity={0.1} />
                    <XAxis dataKey='metric' tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'currentColor', opacity: 0.6 }} />
                    <Tooltip contentStyle={{ backgroundColor: 'oklch(0.18 0.004 285.823)', borderColor: 'oklch(0.27 0.005 286.033)', borderRadius: '8px', fontSize: '12px' }} />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Bar dataKey='crossModal' name='Cross-Modal (SAR → Optical)' fill='#FBBA72' radius={[4, 4, 0, 0]} />
                    <Bar dataKey='sameModal' name='Same-Modal (Optical → Optical)' fill='#34d399' radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* SOTA Peer Model Comparison Table */}
          <Card className='border-border/60'>
            <CardContent className='p-6 space-y-4'>
              <div>
                <h2 className='text-base font-bold text-foreground font-sans'>BEN-14K SOTA Peer Model Comparison Table</h2>
                <p className='text-xs text-muted-foreground font-sans'>Official benchmark comparison on 80/20 non-synthetic BEN-14K partition against baseline literature</p>
              </div>

              <div className='overflow-x-auto rounded-xl border border-border/50'>
                <table className='w-full text-left text-xs font-sans border-collapse'>
                  <thead>
                    <tr className='bg-muted/40 border-b border-border/60 text-muted-foreground font-semibold uppercase tracking-wider text-[10px]'>
                      <th className='p-3'>Model / Paradigm</th>
                      <th className='p-3'>Baseline Type</th>
                      <th className='p-3 text-center'>S1 → S2 F1@5</th>
                      <th className='p-3 text-center'>S2 → S1 F1@5</th>
                      <th className='p-3 text-center'>S1 → S1 F1@5</th>
                      <th className='p-3 text-center'>S2 → S2 F1@5</th>
                      <th className='p-3 text-center'>Cross-Modal mAP</th>
                      <th className='p-3 text-right'>Trainable Params</th>
                    </tr>
                  </thead>
                  <tbody className='divide-y divide-border/40'>
                    {peerComparisonTable.map((row, idx) => {
                      const isOurs = row.model.includes('SABER')
                      return (
                        <tr
                          key={idx}
                          className={cn(
                            'transition-colors hover:bg-muted/30',
                            isOurs ? 'bg-[#FBBA72]/10 font-semibold' : ''
                          )}
                        >
                          <td className='p-3 font-bold text-foreground flex items-center gap-1.5'>
                            {isOurs && <Badge className='bg-[#FBBA72] text-black font-bold text-[9px] px-1 py-0'>OURS</Badge>}
                            {row.model}
                          </td>
                          <td className='p-3 text-muted-foreground'>{row.baseline}</td>
                          <td className={cn('p-3 text-center font-mono', isOurs ? 'text-[#FBBA72] font-bold' : '')}>{row.s1s2}</td>
                          <td className={cn('p-3 text-center font-mono', isOurs ? 'text-[#FBBA72] font-bold' : '')}>{row.s2s1}</td>
                          <td className='p-3 text-center font-mono text-muted-foreground'>{row.s1s1}</td>
                          <td className='p-3 text-center font-mono text-muted-foreground'>{row.s2s2}</td>
                          <td className={cn('p-3 text-center font-mono', isOurs ? 'text-emerald-400 font-bold' : 'text-emerald-500/80')}>{row.map}</td>
                          <td className={cn('p-3 text-right font-mono', isOurs ? 'text-[#FBBA72] font-bold' : 'text-muted-foreground')}>{row.params}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
