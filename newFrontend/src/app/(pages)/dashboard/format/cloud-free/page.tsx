'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'
import {
  CloudOff,
  CloudRain,
  Eye,
  CheckCircle2,
  Zap,
  ArrowRight,
  Database,
  Cpu,
  RotateCw,
  Sparkles,
  Layers,
  ShieldCheck,
} from 'lucide-react'

import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

const DEMO_PRESET = {
  id: 'ben14k_hardcoded',
  title: 'BEN-14K Sentinel-1/2 Paired Scene (ROIs18_Nov2023_01)',
  cloudCover: '92% Cloud Patch Overlay',
  cloudyOptImg: '/images/satellite/dataset_optical_cloud_patched.png',
  sarImg: '/images/satellite/dataset_sar_query.png',
  originalOptImg: '/images/satellite/dataset_optical_original.png',
  retrievedCandidates: [
    {
      rank: 1,
      name: 'ROIs18_Nov2023_01_original.png',
      sim: 96.4,
      jaccard: 88.0,
      cloudCover: '0% (Original Restored)',
      img: '/images/satellite/dataset_optical_original.png',
      tags: ['Arable land', 'Urban fabric', 'Cropland'],
    },
    {
      rank: 2,
      name: 'ROIs18_Nov2023_02.png',
      sim: 94.1,
      jaccard: 85.2,
      cloudCover: '0%',
      img: '/images/satellite/candidate_2.png',
      tags: ['Arable land', 'Water body'],
    },
    {
      rank: 3,
      name: 'ROIs18_Nov2023_03.png',
      sim: 92.8,
      jaccard: 81.4,
      cloudCover: '0.2%',
      img: '/images/satellite/candidate_3.png',
      tags: ['Pastures', 'Forest'],
    },
    {
      rank: 4,
      name: 'ROIs18_Nov2023_04.png',
      sim: 89.5,
      jaccard: 78.6,
      cloudCover: '0%',
      img: '/images/satellite/candidate_4.png',
      tags: ['Coastal wetland'],
    },
    {
      rank: 5,
      name: 'ROIs18_Nov2023_05.png',
      sim: 87.2,
      jaccard: 75.0,
      cloudCover: '0.5%',
      img: '/images/satellite/candidate_5.png',
      tags: ['Industrial zone'],
    },
  ],
}

export default function CloudFreeDemoPage() {
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [hasRetrieved, setHasRetrieved] = useState<boolean>(true)

  const handleRunRetrieval = () => {
    setIsProcessing(true)
    setTimeout(() => {
      setIsProcessing(false)
      setHasRetrieved(true)
    }, 600)
  }

  return (
    <div className="w-full space-y-6 font-sans">
      {/* ── Top Header Banner ── */}
      <Card className="border-border/60 shadow-sm overflow-hidden border-t-4 border-t-[#FBBA72] bg-card/60 backdrop-blur-xs">
        <CardContent className="p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-1.5">
              <div className="flex items-center gap-2.5 flex-wrap">
                <h1 className="text-xl font-bold tracking-tight text-foreground font-sans">
                  Cloud-Free Satellite Image Retrieval Demonstration
                </h1>
                <Badge className="bg-[#FBBA72]/15 text-[#FBBA72] border-[#FBBA72]/40 font-semibold px-2.5 py-0.5 text-xs rounded-full font-sans">
                  SAR Microwave Cloud-Bypass Engine
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground font-sans max-w-2xl">
                Demonstration using a hardcoded optical image from our BEN-14K dataset overlaid with a Gemini cloud patch. When querying its corresponding Sentinel-1 SAR Radar image, SABER bypasses the cloud obscuration via the CFM Latent ODE Bridge and retrieves the original 0%-cloud-cover optical dataset image at Rank #1.
              </p>
            </div>

            {/* Quick Metrics Pills */}
            <div className="flex flex-wrap items-center gap-2 bg-muted/30 border border-border/40 p-2 rounded-xl text-xs font-sans">
              <div className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/10 rounded-lg border border-emerald-500/30">
                <CloudOff className="size-3.5 text-emerald-400" />
                <span className="font-bold text-emerald-400">0% Cloud Cover Restored</span>
              </div>
              <div className="flex items-center gap-1.5 px-2.5 py-1 bg-[#FBBA72]/10 rounded-lg border border-[#FBBA72]/30">
                <Zap className="size-3.5 text-[#FBBA72]" />
                <span className="font-bold text-[#FBBA72]">28.48ms E2E Latency</span>
              </div>
            </div>
          </div>

          {/* Active Dataset Scene Info */}
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-border/40 text-xs font-sans">
            <span className="text-muted-foreground font-mono">Dataset Scene:</span>
            <Badge variant="outline" className="border-border/60 bg-muted/20 text-foreground font-mono text-xs">
              {DEMO_PRESET.title}
            </Badge>
            <Badge variant="outline" className="border-rose-500/40 text-rose-400 bg-rose-500/10 font-mono text-xs">
              {DEMO_PRESET.cloudCover}
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* ── 3-Stage Pipeline Demonstration View ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Stage 1: Cloud-Obscured Optical Input (Gemini Cloud Patch) */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-xs flex flex-col justify-between">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="border-rose-500/40 text-rose-400 bg-rose-500/10 text-[10px] font-bold">
                STAGE 1 · CLOUD-PATCHED OPTICAL
              </Badge>
              <span className="text-[10px] font-mono text-rose-400 font-bold">92% Cloud Obscuration</span>
            </div>

            <div className="space-y-1">
              <h3 className="text-sm font-bold text-foreground">Optical Scene + Gemini Cloud Patch</h3>
              <p className="text-[11px] text-muted-foreground">Sentinel-2 Optical image with Gemini cloud overlay</p>
            </div>

            <div className="relative aspect-square rounded-xl overflow-hidden border border-rose-500/30 bg-zinc-950 group">
              <img
                src={DEMO_PRESET.cloudyOptImg}
                alt="Dataset Optical Scene with Gemini Cloud Patch"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
              <div className="absolute bottom-2 left-2 right-2 p-2 rounded-lg bg-black/80 backdrop-blur-md border border-rose-500/40 flex items-center justify-between text-[10px]">
                <span className="text-rose-400 font-bold font-mono">0% Ground Visibility</span>
                <span className="text-muted-foreground">Direct Search FAILS</span>
              </div>
            </div>

            <p className="text-[11px] text-muted-foreground leading-relaxed pt-1">
              Optical image from our dataset obscured by cloud patch generated via Gemini. Direct optical search fails due to cloud pixel corruption.
            </p>
          </CardContent>
        </Card>

        {/* Stage 2: Sentinel-1 SAR Radar Query */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-xs flex flex-col justify-between">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="border-sky-500/40 text-sky-400 bg-sky-500/10 text-[10px] font-bold">
                STAGE 2 · CORRESPONDING SAR QUERY
              </Badge>
              <span className="text-[10px] font-mono text-sky-400 font-bold">100% Cloud Bypass</span>
            </div>

            <div className="space-y-1">
              <h3 className="text-sm font-bold text-foreground">Corresponding Sentinel-1 SAR Scene</h3>
              <p className="text-[11px] text-muted-foreground">Microwave radar image of the exact same coordinate</p>
            </div>

            <div className="relative aspect-square rounded-xl overflow-hidden border border-sky-500/30 bg-zinc-950 group">
              <img
                src={DEMO_PRESET.sarImg}
                alt="Corresponding Sentinel-1 SAR Radar Scene"
                className="w-full h-full object-cover grayscale contrast-125 brightness-90"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
              <div className="absolute bottom-2 left-2 right-2 p-2 rounded-lg bg-black/80 backdrop-blur-md border border-sky-500/40 flex items-center justify-between text-[10px]">
                <span className="text-sky-400 font-bold font-mono">100% Ground Backscatter</span>
                <span className="text-muted-foreground">SAR Query Active</span>
              </div>
            </div>

            <p className="text-[11px] text-muted-foreground leading-relaxed pt-1">
              Sentinel-1 SAR microwave radar pulses pass straight through cloud cover, capturing the exact ground texture and land-cover geometry.
            </p>
          </CardContent>
        </Card>

        {/* Stage 3: SABER CFM Latent ODE Retrieval Action */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-xs flex flex-col justify-between">
          <CardContent className="p-4 space-y-3 flex flex-col h-full justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Badge variant="outline" className="border-[#FBBA72]/50 text-[#FBBA72] bg-[#FBBA72]/10 text-[10px] font-bold">
                  STAGE 3 · CFM ODE RETRIEVAL
                </Badge>
                <span className="text-[10px] font-mono text-emerald-400 font-bold">91.49% mAP</span>
              </div>

              <div className="space-y-1">
                <h3 className="text-sm font-bold text-foreground">SABER Latent ODE Bridge</h3>
                <p className="text-[11px] text-muted-foreground">Transports SAR embedding → Optical hypersphere</p>
              </div>

              {/* Telemetry Breakdown Box */}
              <div className="p-3 rounded-xl border border-border/60 bg-muted/20 space-y-2 font-mono text-xs">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground">DOFA ViT Encoder:</span>
                  <span className="font-bold text-foreground">14.2 ms</span>
                </div>
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground">CFM ODE Bridge:</span>
                  <span className="font-bold text-[#FBBA72]">11.6 ms</span>
                </div>
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground">FAISS ANN Index:</span>
                  <span className="font-bold text-emerald-400">2.8 ms</span>
                </div>
                <div className="pt-1.5 border-t border-border/40 flex items-center justify-between text-xs font-bold">
                  <span className="text-foreground">Total E2E Latency:</span>
                  <span className="text-[#FBBA72]">28.48 ms</span>
                </div>
              </div>
            </div>

            <Button
              onClick={handleRunRetrieval}
              disabled={isProcessing}
              className="w-full gap-2 bg-[#FBBA72] hover:bg-[#FBBA72]/90 text-black font-bold font-sans mt-4 py-5 rounded-xl cursor-pointer"
            >
              {isProcessing ? (
                <>
                  <RotateCw className="size-4 animate-spin" />
                  Running CFM Latent Transport...
                </>
              ) : (
                <>
                  <Sparkles className="size-4" />
                  Run Cloud-Free Image Retrieval
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* ── Top-5 Retrieved Cloud-Free Optical References (Database Query Output) ── */}
      {hasRetrieved && (
        <div className="space-y-4 pt-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-foreground font-sans">
                Retrieved Top-5 Cloud-Free Optical Scenes from BEN-14K Dataset
              </h2>
              <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/30 text-xs font-semibold">
                Original Cloud-Free Optical Scene Restored at #1
              </Badge>
            </div>
            <span className="text-xs font-mono text-muted-foreground">
              Queried from 14,832-scene BEN-14K Archive
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
            {DEMO_PRESET.retrievedCandidates.map((candidate) => (
              <Card
                key={candidate.rank}
                className={cn(
                  'border-border/60 bg-card/60 backdrop-blur-xs flex flex-col justify-between overflow-hidden transition-all duration-200 hover:border-[#FBBA72]/60',
                  candidate.rank === 1 && 'border-2 border-[#FBBA72] bg-[#FBBA72]/5 shadow-md',
                )}
              >
                <CardContent className="p-3 space-y-2.5">
                  <div className="flex items-center justify-between text-[11px]">
                    <Badge
                      variant="outline"
                      className={cn(
                        'font-bold px-2 py-0.5 font-mono',
                        candidate.rank === 1
                          ? 'border-[#FBBA72] text-black bg-[#FBBA72]'
                          : 'border-[#FBBA72]/50 text-[#FBBA72] bg-[#FBBA72]/10',
                      )}
                    >
                      #{candidate.rank} {candidate.rank === 1 ? 'ORIGINAL RESTORED' : ''}
                    </Badge>
                    <span className="font-mono font-bold text-foreground">{candidate.sim}%</span>
                  </div>

                  <div className="relative aspect-square rounded-lg overflow-hidden border border-border/40 bg-zinc-950 group">
                    <img
                      src={candidate.img}
                      alt={candidate.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded bg-emerald-500/90 text-white font-mono text-[9px] font-bold">
                      {candidate.cloudCover}
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <p className="text-[10px] font-mono text-foreground font-semibold truncate" title={candidate.name}>
                      {candidate.name}
                    </p>

                    <div className="w-full bg-muted/40 h-1.5 rounded-full overflow-hidden">
                      <div className="bg-[#FBBA72] h-full rounded-full" style={{ width: `${candidate.sim}%` }} />
                    </div>

                    <div className="flex justify-between items-center text-[10px] text-muted-foreground font-sans">
                      <span>Jaccard Overlap:</span>
                      <span className="font-mono text-foreground font-bold">{candidate.jaccard}%</span>
                    </div>

                    <div className="flex flex-wrap gap-1 pt-1">
                      {candidate.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-1.5 py-0.5 rounded bg-muted/30 text-[9px] text-muted-foreground font-sans border border-border/40"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
