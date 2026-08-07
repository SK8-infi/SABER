'use client'

import React from 'react'
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
              <img
                src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Floods_in_Valencia_ESA503179_-_Floods_in_Valencia.jpg/960px-Floods_in_Valencia_ESA503179_-_Floods_in_Valencia.jpg"
                alt="Query Scene"
                className="w-full h-full object-cover grayscale contrast-125 brightness-90"
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
                { rank: 1, name: "ROIs18_Nov2023_01", sim: 96.4, jaccard: 88.0, img: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Spain_from_Sentinel-2.jpg/960px-Spain_from_Sentinel-2.jpg" },
                { rank: 2, name: "ROIs18_Nov2023_02", sim: 94.1, jaccard: 85.2, img: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Central-eastern_Brazil%2C_by_Copernicus_Sentinel-2A_satellite.jpg/640px-Central-eastern_Brazil%2C_by_Copernicus_Sentinel-2A_satellite.jpg" },
                { rank: 3, name: "ROIs18_Nov2023_03", sim: 92.8, jaccard: 81.4, img: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Toronto_by_Sentinel-2.jpg/960px-Toronto_by_Sentinel-2.jpg" },
                { rank: 4, name: "ROIs18_Nov2023_04", sim: 89.5, jaccard: 78.6, img: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/North-Sentinel-Island-Sentinel-2A.png/960px-North-Sentinel-Island-Sentinel-2A.png" },
                { rank: 5, name: "ROIs18_Nov2023_05", sim: 87.2, jaccard: 75.0, img: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Berlin_by_Senitnel-2.jpg/960px-Berlin_by_Senitnel-2.jpg" },
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
                    <img src={c.img} alt={c.name} className="w-full h-full object-cover" />
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
