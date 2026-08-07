'use client'

import React, { useState, useRef } from 'react'
import {
  Upload,
  Image as ImageIcon,
  Zap,
  RotateCw,
  CheckCircle2,
  Sliders,
  Sparkles,
  Layers,
  Search,
  X,
  FileCheck,
  Activity,
  Cpu,
  Database,
  Eye,
} from 'lucide-react'

import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import MultiSensorInspector from '@/components/shared/MultiSensorInspector'
import type { SceneData } from '@/views/apps/users/view/user-view-left-panel'

export interface UploadedQueryResultCandidate {
  rank: number
  name: string
  img: string
  similarity: number
  jaccard: number
  modality: 'SAR' | 'Optical'
  tags: string[]
}

export interface CustomImageQueryUploadProps {
  className?: string
}

export default function CustomImageQueryUpload({ className }: CustomImageQueryUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [modality, setModality] = useState<'s1' | 's2'>('s1')
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [hasResults, setHasResults] = useState<boolean>(false)

  // Inspection Modal State
  const [inspectCandidate, setInspectCandidate] = useState<UploadedQueryResultCandidate | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      setHasResults(false)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) {
      setSelectedFile(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      setHasResults(false)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const handleClearFile = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setHasResults(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleRunRetrieval = () => {
    if (!previewUrl) return
    setIsProcessing(true)
    setTimeout(() => {
      setIsProcessing(false)
      setHasResults(true)
    }, 700)
  }

  // Pre-configured satellite candidate matches for uploaded query simulation
  const sameModalityCandidates: UploadedQueryResultCandidate[] = modality === 's1'
    ? [
        { rank: 1, name: 'S1A_IW_GRDH_Matched_01.png', img: '/images/satellite/ben14k_real_sar.png', similarity: 96.8, jaccard: 89.2, modality: 'SAR', tags: ['Urban fabric', 'Arable land'] },
        { rank: 2, name: 'S1A_IW_GRDH_Matched_02.png', img: '/images/satellite/usecase_crop_sar.png', similarity: 94.2, jaccard: 85.0, modality: 'SAR', tags: ['Arable land', 'Water body'] },
        { rank: 3, name: 'S1A_IW_GRDH_Matched_03.png', img: '/images/satellite/usecase_urban_sar.png', similarity: 92.5, jaccard: 82.1, modality: 'SAR', tags: ['Pastures', 'Forest'] },
        { rank: 4, name: 'S1A_IW_GRDH_Matched_04.png', img: '/images/satellite/usecase_defense_sar.png', similarity: 89.1, jaccard: 78.4, modality: 'SAR', tags: ['Coastal wetland'] },
        { rank: 5, name: 'S1A_IW_GRDH_Matched_05.png', img: '/images/satellite/usecase_archive_sar.png', similarity: 87.0, jaccard: 74.8, modality: 'SAR', tags: ['Industrial zone'] },
      ]
    : [
        { rank: 1, name: 'S2A_MSIL2A_Matched_01.png', img: '/images/satellite/ben14k_real_optical.png', similarity: 97.4, jaccard: 91.0, modality: 'Optical', tags: ['Arable land', 'Urban fabric'] },
        { rank: 2, name: 'S2A_MSIL2A_Matched_02.png', img: '/images/satellite/candidate_1.png', similarity: 95.1, jaccard: 86.8, modality: 'Optical', tags: ['Cropland', 'Vegetation'] },
        { rank: 3, name: 'S2A_MSIL2A_Matched_03.png', img: '/images/satellite/candidate_2.png', similarity: 93.0, jaccard: 83.5, modality: 'Optical', tags: ['Water body'] },
        { rank: 4, name: 'S2A_MSIL2A_Matched_04.png', img: '/images/satellite/candidate_3.png', similarity: 90.2, jaccard: 79.2, modality: 'Optical', tags: ['Pastures'] },
        { rank: 5, name: 'S2A_MSIL2A_Matched_05.png', img: '/images/satellite/candidate_4.png', similarity: 87.6, jaccard: 76.1, modality: 'Optical', tags: ['Coastal wetland'] },
      ]

  const crossModalityCandidates: UploadedQueryResultCandidate[] = modality === 's1'
    ? [
        { rank: 1, name: 'S2A_MSIL2A_CrossMatch_01.png', img: '/images/satellite/ben14k_real_optical.png', similarity: 96.4, jaccard: 88.0, modality: 'Optical', tags: ['Arable land', 'Urban fabric'] },
        { rank: 2, name: 'S2A_MSIL2A_CrossMatch_02.png', img: '/images/satellite/candidate_1.png', similarity: 94.1, jaccard: 85.2, modality: 'Optical', tags: ['Cropland', 'Agriculture'] },
        { rank: 3, name: 'S2A_MSIL2A_CrossMatch_03.png', img: '/images/satellite/candidate_2.png', similarity: 92.8, jaccard: 81.4, modality: 'Optical', tags: ['Water body'] },
        { rank: 4, name: 'S2A_MSIL2A_CrossMatch_04.png', img: '/images/satellite/candidate_3.png', similarity: 89.5, jaccard: 78.6, modality: 'Optical', tags: ['Pastures', 'Forest'] },
        { rank: 5, name: 'S2A_MSIL2A_CrossMatch_05.png', img: '/images/satellite/candidate_4.png', similarity: 87.2, jaccard: 75.0, modality: 'Optical', tags: ['Coastal wetland'] },
      ]
    : [
        { rank: 1, name: 'S1A_IW_GRDH_CrossMatch_01.png', img: '/images/satellite/ben14k_real_sar.png', similarity: 95.9, jaccard: 87.5, modality: 'SAR', tags: ['Urban fabric', 'Arable land'] },
        { rank: 2, name: 'S1A_IW_GRDH_CrossMatch_02.png', img: '/images/satellite/usecase_crop_sar.png', similarity: 93.6, jaccard: 84.1, modality: 'SAR', tags: ['Cropland'] },
        { rank: 3, name: 'S1A_IW_GRDH_CrossMatch_03.png', img: '/images/satellite/usecase_urban_sar.png', similarity: 91.8, jaccard: 80.9, modality: 'SAR', tags: ['Urban backscatter'] },
        { rank: 4, name: 'S1A_IW_GRDH_CrossMatch_04.png', img: '/images/satellite/usecase_defense_sar.png', similarity: 88.7, jaccard: 77.2, modality: 'SAR', tags: ['Coastal radar'] },
        { rank: 5, name: 'S1A_IW_GRDH_CrossMatch_05.png', img: '/images/satellite/usecase_archive_sar.png', similarity: 86.4, jaccard: 73.9, modality: 'SAR', tags: ['Industrial zone'] },
      ]

  const candidateToSceneData = (c: UploadedQueryResultCandidate): SceneData => ({
    sceneId: c.name,
    progressColor: 'bg-[#FBBA72]',
    progressWidth: `w-[${Math.round(c.similarity)}%]`,
    jaccard: `${c.jaccard}%`,
    rank: `#${c.rank}`,
    matched: c.jaccard > 70,
    matchScore: `${c.similarity}%`,
    tags: c.tags.map((t) => ({ label: t, checked: true })),
  })

  return (
    <div className={cn('w-full space-y-6 font-sans', className)}>
      {/* Upload & Modality Input Card */}
      <Card className="border-border/60 bg-card/60 backdrop-blur-xs shadow-sm overflow-hidden border-t-4 border-t-[#FBBA72]">
        <CardContent className="p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Upload className="size-4 text-[#FBBA72]" />
                <h3 className="text-base font-bold text-foreground">
                  Custom Satellite Query Upload
                </h3>
              </div>
              <p className="text-xs text-muted-foreground">
                Upload your own Sentinel-1 SAR or Sentinel-2 Optical satellite image file to generate real-time same-modality and cross-modality retrieval.
              </p>
            </div>

            {/* Modality Selector Chips */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider font-mono">
                Query Modality:
              </span>
              <button
                onClick={() => {
                  setModality('s1')
                  setHasResults(false)
                }}
                className={cn(
                  'px-3 py-1 rounded-full text-xs font-semibold border transition-all cursor-pointer font-sans',
                  modality === 's1'
                    ? 'border-sky-500/60 bg-sky-500/15 text-sky-400 font-bold'
                    : 'border-border/60 bg-muted/20 text-muted-foreground hover:text-foreground',
                )}
              >
                Sentinel-1 SAR Radar
              </button>
              <button
                onClick={() => {
                  setModality('s2')
                  setHasResults(false)
                }}
                className={cn(
                  'px-3 py-1 rounded-full text-xs font-semibold border transition-all cursor-pointer font-sans',
                  modality === 's2'
                    ? 'border-[#FBBA72]/60 bg-[#FBBA72]/15 text-[#FBBA72] font-bold'
                    : 'border-border/60 bg-muted/20 text-muted-foreground hover:text-foreground',
                )}
              >
                Sentinel-2 Optical
              </button>
            </div>
          </div>

          {/* Drag & Drop Zone / Preview Grid */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                'md:col-span-8 rounded-2xl border-2 border-dashed p-6 text-center transition-all cursor-pointer flex flex-col items-center justify-center gap-3 bg-muted/10 min-h-[160px]',
                previewUrl
                  ? 'border-[#FBBA72]/60 bg-[#FBBA72]/5'
                  : 'border-border/60 hover:border-[#FBBA72]/40 hover:bg-muted/20',
              )}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.tif,.tiff"
                onChange={handleFileChange}
                className="hidden"
              />

              {previewUrl ? (
                <div className="flex items-center gap-4 text-left w-full">
                  <div className="relative size-24 rounded-xl overflow-hidden border border-border/60 bg-zinc-950 shrink-0 group">
                    <img
                      src={previewUrl}
                      alt="Uploaded Query Preview"
                      className="w-full h-full object-cover"
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleClearFile()
                      }}
                      className="absolute top-1 right-1 p-1 rounded-full bg-black/80 text-white hover:bg-rose-500 transition-colors"
                      title="Remove image"
                    >
                      <X className="size-3" />
                    </button>
                  </div>

                  <div className="space-y-1 flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <FileCheck className="size-4 text-emerald-400" />
                      <span className="font-bold text-sm text-foreground truncate font-mono">
                        {selectedFile?.name ?? 'uploaded_satellite_query.png'}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground font-sans">
                      Modality: <span className="font-semibold text-[#FBBA72]">{modality === 's1' ? 'Sentinel-1 SAR Radar' : 'Sentinel-2 Optical'}</span> · File Size: {selectedFile ? (selectedFile.size / 1024).toFixed(1) : '256'} KB
                    </p>
                    <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 bg-emerald-500/10 text-[10px] font-mono">
                      Ready for SABER CFM Retrieval
                    </Badge>
                  </div>
                </div>
              ) : (
                <>
                  <div className="size-10 rounded-full bg-[#FBBA72]/15 border border-[#FBBA72]/30 flex items-center justify-center text-[#FBBA72]">
                    <Upload className="size-5" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs font-bold text-foreground">
                      Click to upload or drag & drop satellite query image
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      Supports PNG, JPG, WEBP, TIFF satellite image rasters up to 50MB
                    </p>
                  </div>
                </>
              )}
            </div>

            {/* Run Button Column */}
            <div className="md:col-span-4 flex flex-col gap-2">
              <Button
                onClick={handleRunRetrieval}
                disabled={!previewUrl || isProcessing}
                className="w-full py-6 rounded-xl bg-[#FBBA72] hover:bg-[#FBBA72]/90 text-black font-bold font-sans gap-2 text-sm shadow-sm cursor-pointer disabled:opacity-50"
              >
                {isProcessing ? (
                  <>
                    <RotateCw className="size-4 animate-spin" />
                    Processing CFM Retrieval...
                  </>
                ) : (
                  <>
                    <Sparkles className="size-4" />
                    Run Dual-Modal Retrieval
                  </>
                )}
              </Button>
              <p className="text-[10px] text-muted-foreground text-center font-mono">
                Sub-28.5ms latency · DOFA ViT + CFM ODE Bridge
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results View (Same-Modality & Cross-Modality Top-5 Galleries) */}
      {hasResults && (
        <div className="space-y-6 pt-2">
          {/* Telemetry Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: 'DOFA ViT Feature Extraction', val: '14.2 ms', icon: Cpu },
              { label: 'CFM Latent ODE Transport', val: '11.6 ms', icon: Zap },
              { label: 'FAISS Vector Search', val: '2.8 ms', icon: Search },
              { label: 'Total Ingestion & Retrieval', val: '28.48 ms', icon: Activity },
            ].map((t) => (
              <Card key={t.label} className="border-border/60 bg-card/60 backdrop-blur-xs p-3">
                <CardContent className="p-0 flex items-center gap-2.5">
                  <div className="size-8 rounded-lg bg-[#FBBA72]/15 border border-[#FBBA72]/30 flex items-center justify-center text-[#FBBA72] shrink-0">
                    <t.icon className="size-4" />
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="text-[10px] text-muted-foreground uppercase font-mono tracking-wider truncate">
                      {t.label}
                    </span>
                    <span className="text-xs font-bold font-mono text-[#FBBA72]">
                      {t.val}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Section 1: Same-Modality Retrieval Results */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-foreground">
                  Same-Modality Retrieval Results ({modality === 's1' ? 'SAR → SAR' : 'Optical → Optical'})
                </h3>
                <Badge variant="outline" className="border-sky-500/40 text-sky-400 bg-sky-500/10 text-[10px] font-mono">
                  Direct Latent Space Match
                </Badge>
              </div>
              <span className="text-xs font-mono text-emerald-400 font-bold">96.8% Top-1 Similarity</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
              {sameModalityCandidates.map((candidate) => (
                <Card
                  key={candidate.rank}
                  onClick={() => setInspectCandidate(candidate)}
                  className="border-border/60 bg-card/60 backdrop-blur-xs overflow-hidden transition-all duration-200 hover:border-[#FBBA72]/60 cursor-pointer group"
                >
                  <CardContent className="p-3 space-y-2">
                    <div className="flex items-center justify-between text-[11px]">
                      <Badge variant="outline" className="border-sky-500/40 text-sky-400 bg-sky-500/10 font-bold font-mono px-2 py-0.2">
                        #{candidate.rank}
                      </Badge>
                      <span className="font-mono font-bold text-foreground">{candidate.similarity}%</span>
                    </div>

                    <div className="relative aspect-square rounded-lg overflow-hidden border border-border/40 bg-zinc-950">
                      <img
                        src={candidate.img}
                        alt={candidate.name}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <span className="px-2 py-1 rounded bg-[#FBBA72] text-black text-[10px] font-bold flex items-center gap-1 font-sans">
                          <Eye className="size-3" /> Inspect Scene
                        </span>
                      </div>
                    </div>

                    <div className="space-y-1">
                      <p className="text-[10px] font-mono text-foreground font-semibold truncate">{candidate.name}</p>
                      <div className="w-full bg-muted/40 h-1 rounded-full overflow-hidden">
                        <div className="bg-sky-400 h-full rounded-full" style={{ width: `${candidate.similarity}%` }} />
                      </div>
                      <div className="flex justify-between text-[9px] text-muted-foreground font-sans">
                        <span>Jaccard:</span>
                        <span className="font-mono text-foreground font-bold">{candidate.jaccard}%</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Section 2: Cross-Modality Retrieval Results */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-foreground">
                  Cross-Modality Retrieval Results ({modality === 's1' ? 'SAR → Optical via CFM ODE' : 'Optical → SAR via CFM ODE'})
                </h3>
                <Badge variant="outline" className="border-[#FBBA72]/50 text-[#FBBA72] bg-[#FBBA72]/10 text-[10px] font-mono">
                  CFM Latent ODE Transported
                </Badge>
              </div>
              <span className="text-xs font-mono text-emerald-400 font-bold">91.49% mAP Benchmark</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
              {crossModalityCandidates.map((candidate) => (
                <Card
                  key={candidate.rank}
                  onClick={() => setInspectCandidate(candidate)}
                  className="border-border/60 bg-card/60 backdrop-blur-xs overflow-hidden transition-all duration-200 hover:border-[#FBBA72]/60 cursor-pointer group"
                >
                  <CardContent className="p-3 space-y-2">
                    <div className="flex items-center justify-between text-[11px]">
                      <Badge variant="outline" className="border-[#FBBA72]/50 text-[#FBBA72] bg-[#FBBA72]/10 font-bold font-mono px-2 py-0.2">
                        #{candidate.rank}
                      </Badge>
                      <span className="font-mono font-bold text-foreground">{candidate.similarity}%</span>
                    </div>

                    <div className="relative aspect-square rounded-lg overflow-hidden border border-border/40 bg-zinc-950">
                      <img
                        src={candidate.img}
                        alt={candidate.name}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <span className="px-2 py-1 rounded bg-[#FBBA72] text-black text-[10px] font-bold flex items-center gap-1 font-sans">
                          <Eye className="size-3" /> Inspect Scene
                        </span>
                      </div>
                    </div>

                    <div className="space-y-1">
                      <p className="text-[10px] font-mono text-foreground font-semibold truncate">{candidate.name}</p>
                      <div className="w-full bg-muted/40 h-1 rounded-full overflow-hidden">
                        <div className="bg-[#FBBA72] h-full rounded-full" style={{ width: `${candidate.similarity}%` }} />
                      </div>
                      <div className="flex justify-between text-[9px] text-muted-foreground font-sans">
                        <span>Jaccard:</span>
                        <span className="font-mono text-foreground font-bold">{candidate.jaccard}%</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* MultiSensorInspector Modal for Candidate Details */}
      {inspectCandidate && previewUrl && (
        <MultiSensorInspector
          open={!!inspectCandidate}
          onClose={() => setInspectCandidate(null)}
          query={{
            name: selectedFile?.name ?? 'custom_query_scene',
            source_modality: modality,
            active_classes: inspectCandidate.tags,
            thumbnail: previewUrl,
          }}
          candidate={{
            name: inspectCandidate.name,
            rank: inspectCandidate.rank,
            similarity_score: inspectCandidate.similarity,
            jaccard_overlap: inspectCandidate.jaccard,
            active_classes: inspectCandidate.tags,
            thumbnail: inspectCandidate.img,
          }}
        />
      )}
    </div>
  )
}
