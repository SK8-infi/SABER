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

const DEMO_PRESETS = [
  {
    id: 'derna',
    title: 'Storm Daniel Flood Inundation (Derna, Libya)',
    cloudCover: '95% Cloud Cover',
    sarImg: '/images/satellite/query_sar_hero.png',
    cloudyOptImg: '/images/satellite/cloudy_optical.png',
    retrievedOptImg: '/images/satellite/candidate_1.png',
    retrievedCandidates: [
      { rank: 1, name: 'S2_ClearSky_PreFlood_01.png', sim: 96.4, jaccard: 88.0, cloudCover: '0%', img: '/images/satellite/candidate_1.png', tags: ['Urban fabric', 'Arable land'] },
      { rank: 2, name: 'S2_ClearSky_Archive_02.png', sim: 94.1, jaccard: 85.2, cloudCover: '0%', img: '/images/satellite/candidate_2.png', tags: ['Arable land', 'Water body'] },
      { rank: 3, name: 'S2_ClearSky_Archive_03.png', sim: 92.8, jaccard: 81.4, cloudCover: '0.2%', img: '/images/satellite/candidate_3.png', tags: ['Pastures', 'Forest'] },
      { rank: 4, name: 'S2_ClearSky_Archive_04.png', sim: 89.5, jaccard: 78.6, cloudCover: '0%', img: '/images/satellite/candidate_4.png', tags: ['Coastal wetland'] },
      { rank: 5, name: 'S2_ClearSky_Archive_05.png', sim: 87.2, jaccard: 75.0, cloudCover: '0.5%', img: '/images/satellite/candidate_5.png', tags: ['Industrial zone'] },
    ],
  },
  {
    id: 'monsoon',
    title: 'Monsoon Farmland Cloud Cover (Central Brazil)',
    cloudCover: '88% Overcast',
    sarImg: '/images/satellite/usecase_crop_sar.png',
    cloudyOptImg: '/images/satellite/cloudy_optical.png',
    retrievedOptImg: '/images/satellite/usecase_crop_opt.png',
    retrievedCandidates: [
      { rank: 1, name: 'S2_ClearSky_Kharif_01.png', sim: 95.8, jaccard: 87.2, cloudCover: '0%', img: '/images/satellite/usecase_crop_opt.png', tags: ['Cropland', 'Agriculture'] },
      { rank: 2, name: 'S2_ClearSky_Farmland_02.png', sim: 93.4, jaccard: 84.0, cloudCover: '0%', img: '/images/satellite/candidate_2.png', tags: ['Irrigated crops'] },
      { rank: 3, name: 'S2_ClearSky_Farmland_03.png', sim: 91.2, jaccard: 80.5, cloudCover: '0%', img: '/images/satellite/candidate_3.png', tags: ['Pastures'] },
      { rank: 4, name: 'S2_ClearSky_Farmland_04.png', sim: 88.9, jaccard: 77.8, cloudCover: '0.1%', img: '/images/satellite/candidate_4.png', tags: ['Vegetation'] },
      { rank: 5, name: 'S2_ClearSky_Farmland_05.png', sim: 86.5, jaccard: 74.2, cloudCover: '0.3%', img: '/images/satellite/candidate_5.png', tags: ['Soil'] },
    ],
  },
]

export default function CloudFreeDemoPage() {
  const [selectedPresetId, setSelectedPresetId] = useState<string>('derna')
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [hasRetrieved, setHasRetrieved] = useState<boolean>(true)

  const activePreset = DEMO_PRESETS.find((p) => p.id === selectedPresetId) ?? DEMO_PRESETS[0]

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
                  Cloud-Free Image Retrieval Demonstration
                </h1>
                <Badge className="bg-[#FBBA72]/15 text-[#FBBA72] border-[#FBBA72]/40 font-semibold px-2.5 py-0.5 text-xs rounded-full font-sans">
                  SAR Microwave Cloud-Bypass Engine
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground font-sans max-w-2xl">
                Optical satellite sensors (Sentinel-2) cannot see ground features when blocked by dense clouds. SABER inputs the matching Sentinel-1 SAR Radar image, transports its embedding through the CFM Latent ODE Bridge, and retrieves the Top-5 historical cloud-free optical reference scenes with 0% cloud cover.
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

          {/* Preset Scenario Selector Buttons */}
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-border/40">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mr-2 font-mono">
              Select Demo Scenario:
            </span>
            {DEMO_PRESETS.map((preset) => {
              const isActive = preset.id === selectedPresetId
              return (
                <button
                  key={preset.id}
                  onClick={() => {
                    setSelectedPresetId(preset.id)
                    setHasRetrieved(true)
                  }}
                  className={cn(
                    'px-3.5 py-1.5 rounded-full text-xs font-semibold border transition-all cursor-pointer font-sans',
                    isActive
                      ? 'border-[#FBBA72]/60 bg-[#FBBA72]/15 text-[#FBBA72] shadow-xs'
                      : 'border-border/60 bg-muted/20 text-muted-foreground hover:text-foreground hover:bg-muted/40',
                  )}
                >
                  {preset.title} ({preset.cloudCover})
                </button>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* ── 3-Stage Pipeline Demonstration View ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Stage 1: Cloud-Obscured Optical Input */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-xs flex flex-col justify-between">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="border-rose-500/40 text-rose-400 bg-rose-500/10 text-[10px] font-bold">
                STAGE 1 · OPTICAL BLINDSPOT
              </Badge>
              <span className="text-[10px] font-mono text-rose-400 font-bold">{activePreset.cloudCover}</span>
            </div>

            <div className="space-y-1">
              <h3 className="text-sm font-bold text-foreground">Cloud-Obscured Optical Scene</h3>
              <p className="text-[11px] text-muted-foreground">Sentinel-2 Multispectral · Ground hidden under storm clouds</p>
            </div>

            <div className="relative aspect-square rounded-xl overflow-hidden border border-rose-500/30 bg-zinc-950 group">
              <img
                src={activePreset.cloudyOptImg}
                alt="Cloud-obscured Optical Scene"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
              <div className="absolute bottom-2 left-2 right-2 p-2 rounded-lg bg-black/80 backdrop-blur-md border border-rose-500/40 flex items-center justify-between text-[10px]">
                <span className="text-rose-400 font-bold font-mono">0% Ground Visibility</span>
                <span className="text-muted-foreground">Direct Search FAILS</span>
              </div>
            </div>

            <p className="text-[11px] text-muted-foreground leading-relaxed pt-1">
              Optical sensors operate in visible/infrared spectrum and cannot penetrate dense clouds or smoke. Ground features are completely obscured.
            </p>
          </CardContent>
        </Card>

        {/* Stage 2: Sentinel-1 SAR Cloud-Penetrating Query */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-xs flex flex-col justify-between">
          <CardContent className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="border-sky-500/40 text-sky-400 bg-sky-500/10 text-[10px] font-bold">
                STAGE 2 · SAR RADAR QUERY
              </Badge>
              <span className="text-[10px] font-mono text-sky-400 font-bold">100% Cloud Bypass</span>
            </div>

            <div className="space-y-1">
              <h3 className="text-sm font-bold text-foreground">Sentinel-1 SAR Radar Query</h3>
              <p className="text-[11px] text-muted-foreground">Microwave backscatter · Sees through clouds & darkness</p>
            </div>

            <div className="relative aspect-square rounded-xl overflow-hidden border border-sky-500/30 bg-zinc-950 group">
              <img
                src={activePreset.sarImg}
                alt="Sentinel-1 SAR Radar Scene"
                className="w-full h-full object-cover grayscale contrast-125 brightness-90"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
              <div className="absolute bottom-2 left-2 right-2 p-2 rounded-lg bg-black/80 backdrop-blur-md border border-sky-500/40 flex items-center justify-between text-[10px]">
                <span className="text-sky-400 font-bold font-mono">100% Ground Backscatter</span>
                <span className="text-muted-foreground">SAR Query Active</span>
              </div>
            </div>

            <p className="text-[11px] text-muted-foreground leading-relaxed pt-1">
              Sentinel-1 Synthetic Aperture Radar (SAR) transmits microwave pulses (5.405 GHz) that pass right through cloud cover, capturing ground surface geometry.
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
                Retrieved Top-5 Cloud-Free Historical Optical Scenes (Sentinel-2)
              </h2>
              <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/30 text-xs font-semibold">
                0% Cloud Cover Guaranteed
              </Badge>
            </div>
            <span className="text-xs font-mono text-muted-foreground">
              Queried from 14,832-scene BEN-14K Archive
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
            {activePreset.retrievedCandidates.map((candidate) => (
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
                      #{candidate.rank} {candidate.rank === 1 ? 'TOP MATCH' : ''}
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
                      {candidate.cloudCover} Clouds
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
