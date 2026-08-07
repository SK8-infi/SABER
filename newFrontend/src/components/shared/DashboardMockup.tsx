'use client'

import React, { useState } from 'react'
import {
  Activity,
  Layers,
  RotateCw,
  Search,
  Sliders,
  Sparkles,
  Tag,
  Zap,
  CheckCircle2,
  Database,
  Cpu,
  Eye,
  FileText,
  Share2,
} from 'lucide-react'
import LogoSvg from '@/assets/svg/logo'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

function SyntheticSAR() {
  return (
    <svg viewBox="0 0 120 120" className="w-full h-full bg-zinc-950">
      <defs>
        <linearGradient id="sarGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0f172a" />
          <stop offset="50%" stopColor="#1e293b" />
          <stop offset="100%" stopColor="#090d16" />
        </linearGradient>
      </defs>
      <rect width="120" height="120" fill="url(#sarGrad)" />
      {Array.from({ length: 24 }).map((_, i) => (
        <line
          key={i}
          x1={0}
          y1={i * 5}
          x2={120}
          y2={i * 5 + Math.sin(i) * 4}
          stroke={`rgba(56, 189, 248, ${0.15 + (i % 3) * 0.1})`}
          strokeWidth={1}
        />
      ))}
      {Array.from({ length: 6 }).map((_, i) => (
        <rect
          key={`b${i}`}
          x={15 + i * 16}
          y={20 + (i % 3) * 28}
          width={10 + (i % 3) * 6}
          height={10 + (i % 2) * 8}
          fill="rgba(251, 186, 114, 0.4)"
          rx={2}
        />
      ))}
    </svg>
  )
}

function SyntheticOpticalTile({ variant = 0 }: { variant?: number }) {
  const colors = [
    { bg: "#1e3a29", field1: "#2d5a3f", field2: "#407a52", river: "#0284c7" },
    { bg: "#1c382b", field1: "#346648", field2: "#4a8a62", river: "#0369a1" },
    { bg: "#233d28", field1: "#3a6140", field2: "#52855b", river: "#38bdf8" },
    { bg: "#1b3324", field1: "#2e543b", field2: "#437854", river: "#0284c7" },
    { bg: "#25422f", fill: "#3b6b4a", field2: "#508c62", river: "#0ea5e9" },
  ]
  const c = colors[variant % colors.length]
  return (
    <svg viewBox="0 0 120 120" className="w-full h-full">
      <rect width="120" height="120" fill={c.bg} />
      <rect x="10" y="10" width="45" height="40" fill={c.field1} rx="4" />
      <rect x="60" y="15" width="50" height="35" fill={c.field2} rx="4" />
      <rect x="15" y="55" width="40" height="55" fill={c.field2} rx="4" />
      <rect x="60" y="55" width="50" height="50" fill={c.field1} rx="4" />
      <path d="M0 40 Q 60 70 120 45" stroke={c.river} strokeWidth="6" fill="none" opacity="0.8" />
    </svg>
  )
}

function DatasetImg({
  src,
  alt,
  fallbackVariant = 0,
  className,
}: {
  src: string
  alt: string
  fallbackVariant?: number
  className?: string
}) {
  const [error, setError] = useState(false)
  if (error) {
    return <SyntheticOpticalTile variant={fallbackVariant} />
  }
  return (
    <img
      src={src}
      alt={alt}
      className={cn("w-full h-full object-cover", className)}
      onError={() => setError(true)}
    />
  )
}

export default function DashboardMockup() {
  return (
    <div className="w-full rounded-2xl border border-border/60 bg-background text-foreground shadow-2xl overflow-hidden font-sans text-left text-xs select-none">
      <div className="grid grid-cols-1 md:grid-cols-12 min-h-[580px]">
        {/* ── Left Sidebar Mockup (3 cols) ── */}
        <div className="md:col-span-3 border-r border-border/50 bg-card/60 p-3 flex flex-col gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2.5 px-2 py-1">
            <LogoSvg className="size-6 text-[#FBBA72]" />
            <div className="flex flex-col">
              <span className="font-bold text-sm text-foreground tracking-tight">SABER</span>
              <span className="text-[10px] text-muted-foreground">Scientific Retrieval</span>
            </div>
          </div>

          {/* Format Menu */}
          <div className="space-y-1">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-2">FORMAT</span>
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-[#FBBA72]/15 border border-[#FBBA72]/40 text-[#FBBA72] font-semibold text-xs">
                <FileText className="size-3.5" />
                <span>Query Engine</span>
              </div>
              <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-muted-foreground hover:text-foreground text-xs">
                <Sliders className="size-3.5" />
                <span>Ablation</span>
              </div>
              <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-muted-foreground hover:text-foreground text-xs">
                <Activity className="size-3.5" />
                <span>Training</span>
              </div>
              <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-muted-foreground hover:text-foreground text-xs">
                <Share2 className="size-3.5" />
                <span>Embedding Space</span>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="space-y-3 pt-2 border-t border-border/40 px-1">
            <div className="space-y-1">
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">DATASET</span>
              <div className="p-2 rounded-lg border border-border/60 bg-muted/20 text-xs font-medium text-foreground">
                BEN-14K — Sentinel-1/2
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">SOURCE</span>
              <div className="p-2 rounded-lg border border-border/60 bg-muted/20 text-xs font-medium text-foreground">
                Sentinel-1 SAR (2ch)
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                <span>SCENE INDEX</span>
                <span className="text-[#FBBA72] font-bold">#0</span>
              </div>
              <div className="p-2 rounded-lg border border-border/60 bg-muted/20 text-xs font-mono text-foreground flex justify-between items-center">
                <span>0</span>
                <RotateCw className="size-3 text-muted-foreground" />
              </div>
            </div>
          </div>
        </div>

        {/* ── Main Dashboard Content (9 cols) ── */}
        <div className="md:col-span-9 bg-background/95 p-4 sm:p-5 flex flex-col gap-4">
          {/* Top Bar */}
          <div className="flex items-center justify-between border-b border-border/40 pb-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground font-sans">
              <span>Dashboard</span>
              <span>&gt;</span>
              <span>Format</span>
              <span>&gt;</span>
              <span className="text-foreground font-semibold">Query Engine</span>
            </div>

            {/* Metrics Pills */}
            <div className="flex items-center gap-2 font-mono text-[11px]">
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-[#FBBA72]/40 bg-[#FBBA72]/10 text-[#FBBA72]">
                <Zap className="size-3" />
                <span>LATENCY 28.48ms</span>
              </div>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-border/60 bg-muted/20 text-muted-foreground">
                <Database className="size-3 text-emerald-400" />
                <span>GALLERY 14,832</span>
              </div>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-border/60 bg-muted/20 text-muted-foreground">
                <Cpu className="size-3 text-sky-400" />
                <span>VRAM 294.9KB</span>
              </div>
            </div>
          </div>

          {/* Active Query Scene Card */}
          <div className="p-4 rounded-2xl border border-border/60 bg-card/80 backdrop-blur-md shadow-sm flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <div className="relative size-24 rounded-xl overflow-hidden border border-border/60 bg-zinc-950 shrink-0">
              <DatasetImg
                src="/images/dataset/demo_11_1.png"
                alt="BEN-14K Query Scene"
                fallbackVariant={0}
                className="grayscale contrast-125 brightness-90"
              />
              <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-sky-500/80 text-[9px] font-bold font-mono text-white">S1</span>
            </div>

            <div className="space-y-2 flex-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-mono font-bold text-foreground truncate max-w-[280px]">
                    S2A_MSIL2A_20170803T094031_58_90_paired.png
                  </h3>
                  <p className="text-[11px] text-muted-foreground font-sans">
                    Query scene for multi-sensor dual-modal retrieval
                  </p>
                </div>

                <div className="flex items-center gap-1.5">
                  <Badge variant="outline" className="border-border/60 bg-muted/20 text-[10px] text-foreground">Urban fabric</Badge>
                  <Badge variant="outline" className="border-border/60 bg-muted/20 text-[10px] text-foreground">Arable land</Badge>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-border/40 text-[11px]">
                <div>
                  <span className="text-[10px] uppercase font-bold text-muted-foreground block font-mono">SAME-MODAL</span>
                  <span className="font-semibold text-sky-400 font-sans">Sentinel-1 SAR</span>
                </div>
                <div>
                  <span className="text-[10px] uppercase font-bold text-muted-foreground block font-mono">CROSS-MODAL</span>
                  <span className="font-semibold text-[#FBBA72] font-sans">Sentinel-2 MS (CFM ODE)</span>
                </div>
                <div>
                  <span className="text-[10px] uppercase font-bold text-muted-foreground block font-mono">RETRIEVE</span>
                  <span className="font-semibold text-foreground font-sans">Top-5 Matches</span>
                </div>
              </div>
            </div>
          </div>

          {/* Results Grid Mockup */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h4 className="text-xs font-bold text-foreground font-sans">Cross-Modality Retrieval Results</h4>
                <Badge variant="outline" className="border-[#FBBA72]/50 text-[#FBBA72] bg-[#FBBA72]/10 text-[10px]">
                  S1 SAR → S2 Optical (CFM ODE)
                </Badge>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 font-bold">91.49% mAP benchmark</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {[
                { rank: 1, name: "ROIs18_Nov2023_01", sim: 96.4, jaccard: 88.0, img: "/images/dataset/demo_18_0.png" },
                { rank: 2, name: "ROIs18_Nov2023_02", sim: 94.1, jaccard: 85.2, img: "/images/dataset/demo_4_0.png" },
                { rank: 3, name: "ROIs18_Nov2023_03", sim: 92.8, jaccard: 81.4, img: "/images/dataset/demo_11_1.png" },
                { rank: 4, name: "ROIs18_Nov2023_04", sim: 89.5, jaccard: 78.6, img: "/images/dataset/demo_18_0.png" },
                { rank: 5, name: "ROIs18_Nov2023_05", sim: 87.2, jaccard: 75.0, img: "/images/dataset/demo_4_0.png" },
              ].map((c) => (
                <div
                  key={c.rank}
                  className="p-2.5 rounded-xl border border-border/60 bg-card/60 flex flex-col gap-2 hover:border-[#FBBA72]/50 transition-colors"
                >
                  <div className="flex items-center justify-between text-[10px]">
                    <Badge variant="outline" className="border-[#FBBA72]/50 text-[#FBBA72] bg-[#FBBA72]/10 font-bold px-1.5 py-0.2">
                      #{c.rank}
                    </Badge>
                    <span className="font-mono font-bold text-foreground">{c.sim}%</span>
                  </div>

                  <div className="relative aspect-square rounded-lg overflow-hidden border border-border/40 bg-zinc-950">
                    <DatasetImg
                      src={c.img}
                      alt={c.name}
                      fallbackVariant={c.rank}
                    />
                  </div>

                  <div className="space-y-1">
                    <p className="text-[10px] font-mono text-foreground truncate">{c.name}</p>
                    <div className="w-full bg-muted/40 h-1.5 rounded-full overflow-hidden">
                      <div className="bg-[#FBBA72] h-full rounded-full" style={{ width: `${c.sim}%` }} />
                    </div>
                    <div className="flex justify-between text-[9px] text-muted-foreground font-sans">
                      <span>Jaccard:</span>
                      <span className="font-mono text-foreground font-semibold">{c.jaccard}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
