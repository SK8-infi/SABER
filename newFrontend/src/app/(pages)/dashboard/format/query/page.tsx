'use client'

import { useEffect, useState, useCallback } from 'react'
import {
  ActivityIcon,
  CpuIcon,
  ZapIcon,
  SearchIcon,
  DatabaseIcon,
  CheckCircle2Icon,
  RotateCw,
  AlertCircle,
  LayersIcon,
  ImageIcon,
} from 'lucide-react'

import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { UserViewLeftPanel, type SceneData } from '@/views/apps/users/view/user-view-left-panel'
import { useRetrievalParams } from '@/contexts/retrieval-params-context'

import MultiSensorInspector from '@/components/shared/MultiSensorInspector'

/* ── API types ─────────────────────────────────────────────── */
interface Candidate {
  rank: number
  name: string
  thumbnail: string
  similarity_score: number
  jaccard_overlap: number
  active_classes: string[]
}

interface LatencyTelemetry {
  preprocessing_ms: number | null
  feature_extraction_ms: number | null
  latent_bridge_ms: number | null
  faiss_search_ms: number | null
  rerank_ms: number | null
  total_latency_ms: number | null
}

interface QueryInfo {
  name: string
  index: number
  source_modality: string
  target_modality: string
  label_indices: number[]
  active_classes: string[]
  thumbnail: string
}

interface QueryResult {
  query: QueryInfo
  candidates: Candidate[]
  latency_telemetry: LatencyTelemetry
}

/* ── helpers ────────────────────────────────────────────────── */
function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  return `${v} ms`
}

function candidateToSceneData(c: Candidate, queryClasses: string[]): SceneData {
  const sim = c.similarity_score
  const pct = Math.min(Math.max(Math.round(sim), 0), 100)
  const hasOverlap = c.jaccard_overlap > 0 ||
    c.active_classes.some(cl => queryClasses.includes(cl))

  return {
    sceneId: c.name,
    progressColor: 'bg-[#FBBA72]',
    progressWidth: `w-[${pct}%]`,
    jaccard: `${c.jaccard_overlap}%`,
    rank: `#${c.rank}`,
    matched: hasOverlap,
    matchScore: `${sim}%`,
    tags: c.active_classes.slice(0, 4).map(label => ({
      label,
      checked: queryClasses.includes(label),
    })),
  }
}

/* ── skeleton card ──────────────────────────────────────────── */
function SkeletonCard() {
  return (
    <div className='rounded-xl border border-border/60 bg-muted/10 p-3.5 flex flex-col justify-between gap-2.5 animate-pulse h-full'>
      <div className='w-full aspect-square rounded-lg bg-muted/40 shrink-0' />
      <div className='flex flex-col gap-1.5 shrink-0'>
        <div className='h-3 w-3/4 rounded bg-muted/40' />
        <div className='h-1.5 w-full rounded bg-muted/40' />
      </div>
      <div className='h-3 w-1/2 rounded bg-muted/40 shrink-0' />
      <div className='h-[72px] w-full rounded bg-muted/20 shrink-0' />
      <div className='h-8 w-full rounded-md bg-muted/40 mt-auto shrink-0' />
    </div>
  )
}

/* ── latency metric boxes ───────────────────────────────────── */
function MetricBoxes({ lats }: { lats: LatencyTelemetry }) {
  const items = [
    { icon: <ActivityIcon className='size-3.5 text-[#FBBA72]' />, title: 'PREP',          value: fmtMs(lats.preprocessing_ms) },
    { icon: <CpuIcon      className='size-3.5 text-[#FBBA72]' />, title: 'FEAT EXT',      value: fmtMs(lats.feature_extraction_ms) },
    { icon: <ZapIcon      className='size-3.5 text-[#FBBA72]' />, title: 'CFM ODE',       value: fmtMs(lats.latent_bridge_ms) },
    { icon: <SearchIcon   className='size-3.5 text-[#FBBA72]' />, title: 'FAISS',         value: fmtMs(lats.faiss_search_ms) },
    { icon: <DatabaseIcon className='size-3.5 text-[#FBBA72]' />, title: 'RE-RANK',       value: fmtMs(lats.rerank_ms) },
    { icon: <CheckCircle2Icon className='size-3.5 text-[#FBBA72]' />, title: 'TOTAL',     value: fmtMs(lats.total_latency_ms) },
  ]

  return (
    <div className='grid gap-2 grid-cols-2 sm:grid-cols-3'>
      {items.map((metric, index) => (
        <Card key={index} className='ring-foreground/10 py-1.5 px-2.5 shadow-none ring-1 hover:border-primary/50 transition-colors'>
          <CardContent className='flex items-center gap-2 p-0'>
            <Avatar className='rounded-md after:border-0 size-7 shrink-0'>
              <AvatarFallback className='bg-primary/10 text-primary shrink-0 rounded-md p-1'>
                {metric.icon}
              </AvatarFallback>
            </Avatar>
            <div className='flex flex-col gap-0 min-w-0'>
              <span className='text-muted-foreground text-[9px] font-medium uppercase tracking-wide truncate font-sans'>{metric.title}</span>
              <span className='text-xs font-semibold text-foreground truncate font-sans'>{metric.value}</span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

/* ── skeleton metric boxes ──────────────────────────────────── */
function SkeletonMetricBoxes() {
  return (
    <div className='grid gap-2 grid-cols-2 sm:grid-cols-3 animate-pulse'>
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className='ring-1 ring-foreground/10 rounded-lg py-1.5 px-2.5 flex items-center gap-2'>
          <div className='size-7 rounded-md bg-muted/40' />
          <div className='flex flex-col gap-1 flex-1'>
            <div className='h-2 w-10 rounded bg-muted/40' />
            <div className='h-2.5 w-14 rounded bg-muted/40' />
          </div>
        </div>
      ))}
    </div>
  )
}

/* ── model result section ───────────────────────────────────── */
interface ModelSectionProps {
  label: string
  modelName: string
  logo: string
  logoBg?: string
  result: QueryResult | null
  loading: boolean
  error: string | null
  onRetry: () => void
  topK: number
  onInspectCandidate?: (c: Candidate) => void
}

function ModelSection({ label, modelName, logo, logoBg, result, loading, error, onRetry, topK, onInspectCandidate }: ModelSectionProps) {
  const queryClasses = result?.query.active_classes ?? []

  return (
    <Card>
      <CardContent className='pt-6'>
        <div className='grid gap-6 lg:grid-cols-12 items-start'>
          {/* Left: model identity */}
          <div className='flex flex-col gap-6 lg:col-span-6'>
            <span className='text-lg font-semibold font-sans'>{label}</span>
            <div className='flex items-center gap-3'>
              <img
                src={logo}
                className={`size-10.5 rounded-lg object-contain p-0.5 ${logoBg ?? ''}`}
                alt={modelName}
              />
              <span className='text-xl font-medium font-sans'>{modelName}</span>
            </div>
          </div>

          {/* Right: latency metrics */}
          <div className='lg:col-span-6 self-start pt-1 w-full'>
            {loading && <SkeletonMetricBoxes />}
            {result && !loading && <MetricBoxes lats={result.latency_telemetry} />}
            {error && !loading && (
              <div className='flex items-center gap-2 text-xs text-rose-500 font-sans'>
                <AlertCircle className='size-3.5 shrink-0' />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>
      </CardContent>

      {/* Retrieved scenes */}
      <CardContent className='pt-4 pb-6 w-full space-y-3'>
        <div className='flex items-center justify-between'>
          <span className='text-sm font-semibold text-foreground uppercase tracking-wider font-sans'>Retrieved Scenes</span>
          <span className='text-xs text-muted-foreground font-medium font-sans'>
            {loading ? '…' : result ? `${result.candidates.length} items` : '—'}
          </span>
        </div>

        {error && !loading && (
          <div className='flex flex-col items-center gap-3 py-8'>
            <AlertCircle className='size-8 text-rose-500/60' />
            <p className='text-sm text-muted-foreground font-sans'>{error}</p>
            <Button variant='outline' size='sm' onClick={onRetry} className='gap-2 font-sans'>
              <RotateCw className='size-3.5' /> Retry
            </Button>
          </div>
        )}

        <div className='grid gap-4 grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 items-stretch w-full'>
          {loading && Array.from({ length: topK }).map((_, i) => <SkeletonCard key={i} />)}

          {!loading && result && result.candidates.map((c, i) => (
            <UserViewLeftPanel
              key={`${label}-${c.rank}`}
              user={{
                id: String(i),
                name: c.name,
                avatar: c.thumbnail,
                email: '',
                role: 'Guest' as const,
                plan: 'Basic' as const,
                status: 'Active' as const,
                billing: 'Manual' as const,
                joinedDate: '',
              }}
              sceneData={candidateToSceneData(c, queryClasses)}
              onInspect={() => onInspectCandidate?.(c)}
              className='w-full hover:border-[#FBBA72]/40 transition-colors'
            />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

/* ── main page ──────────────────────────────────────────────── */
export default function QueryPage() {
  const { params, setTelemetry } = useRetrievalParams()
  const { dataset, srcMod, qIdx, topK, bridge, rerank, odeSteps } = params

  const isBen = dataset === 'ben14k'
  const crossTarget = isBen
    ? (srcMod === 's1' ? 's2' : 's1')
    : (srcMod === 'pan' ? 'ms' : 'pan')

  const getModalityLabel = (mod: string) => {
    const labels: Record<string, string> = {
      s1: 'Sentinel-1 SAR', s2: 'Sentinel-2 MS',
      pan: 'Gaofen-1 PAN', ms: 'Gaofen-1 MS',
    }
    return labels[mod] ?? mod.toUpperCase()
  }

  // 2 result slots: saber-same, saber-cross
  const [saberSameResult,  setSaberSameResult]  = useState<QueryResult | null>(null)
  const [saberCrossResult, setSaberCrossResult] = useState<QueryResult | null>(null)

  const [saberSameLoading,  setSaberSameLoading]  = useState(false)
  const [saberCrossLoading, setSaberCrossLoading] = useState(false)

  const [saberSameError,  setSaberSameError]  = useState<string | null>(null)
  const [saberCrossError, setSaberCrossError] = useState<string | null>(null)

  const runQuery = useCallback(async () => {
    setSaberSameLoading(true);  setSaberCrossLoading(true)
    setSaberSameError(null);    setSaberCrossError(null)

    const makeBody = (tgt: string, useBridge: boolean, useRerank: boolean) => ({
      dataset_name:    dataset,
      query_index:     qIdx,
      source_modality: srcMod,
      target_modality: tgt,
      top_k:           topK,
      enable_bridge:   useBridge,
      enable_rerank:   useRerank,
      ode_steps:       odeSteps,
    })

    const doFetch = (body: object) =>
      fetch('/api/retrieval/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

    // Fire SABER Same and SABER Cross in parallel
    const [saberSameRes, saberCrossRes] = await Promise.allSettled([
      doFetch(makeBody(srcMod,  false, false)),       // SABER same-modal
      doFetch(makeBody(crossTarget, bridge, rerank)),     // SABER cross-modal
    ])

    // Process results
    const processResult = async (
      res: PromiseSettledResult<Response>,
      setResult: (r: QueryResult | null) => void,
      setError: (e: string | null) => void,
      setLoading: (l: boolean) => void,
      label: string,
    ) => {
      if (res.status === 'fulfilled') {
        if (res.value.ok) {
          const data = await res.value.json()
          setResult(data)
          if (label === 'SABER Cross' && data?.latency_telemetry?.total_latency_ms) {
            setTelemetry({ total_latency_ms: data.latency_telemetry.total_latency_ms })
          }
        } else {
          setError(`${label}: HTTP ${res.value.status}`)
        }
      } else {
        setError(`${label}: ${res.reason}`)
      }
      setLoading(false)
    }

    await Promise.all([
      processResult(saberSameRes,  setSaberSameResult,  setSaberSameError,  setSaberSameLoading,  'SABER Same'),
      processResult(saberCrossRes, setSaberCrossResult, setSaberCrossError, setSaberCrossLoading, 'SABER Cross'),
    ])
  }, [dataset, qIdx, srcMod, crossTarget, topK, bridge, rerank, odeSteps, setTelemetry])

  useEffect(() => { runQuery() }, [runQuery])

  // Use any available result for the query info card
  const queryInfo = saberSameResult?.query ?? saberCrossResult?.query
  const topLoading = saberSameLoading && saberCrossLoading

  const [inspectorCandidate, setInspectorCandidate] = useState<Candidate | null>(null)

  return (
    <div className='w-full space-y-6'>

      {/* ── Top query image card ── */}
      <Card className='border-border/60 shadow-sm overflow-hidden'>
        <CardContent className='p-3'>
          {/* Query Scene Details */}
          <div className='flex flex-wrap items-center justify-between gap-3 text-sm font-sans border-b border-border/40 pb-3 mb-3'>
            <div className='flex items-center gap-2 font-mono font-semibold text-foreground text-xs'>
              <LayersIcon className='size-4 text-[#FBBA72]' />
              <span>{queryInfo?.name || `sample_${qIdx}`}</span>
            </div>
            <div className='flex items-center gap-2'>
              <Badge variant='outline' className='border-sky-500/40 text-sky-400 bg-sky-500/10 text-xs font-mono font-semibold'>
                Source: {getModalityLabel(srcMod)}
              </Badge>
              <Badge variant='outline' className='border-emerald-500/40 text-emerald-400 bg-emerald-500/10 text-xs font-mono font-semibold'>
                Target: {getModalityLabel(crossTarget)}
              </Badge>
            </div>
          </div>

          <div className='flex flex-col sm:flex-row items-center sm:items-start gap-4 p-1'>
            <div className='relative aspect-square size-36 sm:size-40 rounded-xl border border-border/60 bg-muted/20 overflow-hidden shrink-0 shadow-xs'>
              {queryInfo?.thumbnail ? (
                <img src={queryInfo.thumbnail} alt={queryInfo.name} className='w-full h-full object-cover hover:scale-105 transition-transform duration-300' />
              ) : (
                <div className='w-full h-full flex items-center justify-center text-muted-foreground/30'>
                  <ImageIcon className='size-6' />
                </div>
              )}
            </div>
            <div className='flex-1 space-y-3 font-sans pt-1'>
              <div className='flex flex-wrap items-center gap-2 text-xs'>
                <span className='text-muted-foreground font-semibold uppercase tracking-wide text-[11px]'>Active Land Cover Classes:</span>
                {queryInfo?.active_classes?.map((cls, idx) => (
                  <Badge key={idx} variant='secondary' className='text-[11px] font-medium px-2.5 py-0.5 bg-muted/60 text-foreground border border-border/40'>
                    {cls}
                  </Badge>
                )) ?? <span className='text-muted-foreground italic text-xs'>Loading taxonomy...</span>}
              </div>
              <p className='text-xs text-muted-foreground leading-relaxed'>
                Multi-sensor query scene execution using SABER continuous flow matching (CFM) probability ODE bridge translation and cosine similarity FAISS retrieval.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── SAME-MODAL RETRIEVAL ── */}
      <div className='space-y-1'>
        <h2 className='text-lg font-bold font-sans text-foreground flex items-center gap-2'>
          <span>Same-Modal Retrieval</span>
          <span className='text-xs font-medium text-muted-foreground bg-muted/50 border border-border/50 px-2.5 py-0.5 rounded-full'>
            {srcMod.toUpperCase()} → {srcMod.toUpperCase()}
          </span>
        </h2>
        <p className='text-xs text-muted-foreground font-sans'>Retrieving within the same sensor — no bridge translation needed</p>
      </div>

      <ModelSection
        label='SABER'
        modelName='SABER — Same Sensor'
        logo='/images/brands/logo-square.webp'
        result={saberSameResult}
        loading={saberSameLoading}
        error={saberSameError}
        onRetry={runQuery}
        topK={topK}
        onInspectCandidate={c => setInspectorCandidate(c)}
      />

      {/* ── CROSS-MODAL RETRIEVAL ── */}
      <div className='space-y-1 pt-4'>
        <h2 className='text-lg font-bold font-sans text-foreground flex items-center gap-2'>
          <span>Cross-Modal Retrieval</span>
          <span className='text-xs font-medium text-[#FBBA72] bg-[#FBBA72]/10 border border-[#FBBA72]/30 px-2.5 py-0.5 rounded-full font-semibold'>
            {srcMod.toUpperCase()} → {crossTarget.toUpperCase()}
          </span>
        </h2>
        <p className='text-xs text-muted-foreground font-sans'>Retrieving across sensors — SABER uses CFM Bridge for cross-modal translation</p>
      </div>

      <ModelSection
        label='SABER'
        modelName='SABER — Neural ODE Bridge (Ours)'
        logo='/images/brands/logo-square.webp'
        result={saberCrossResult}
        loading={saberCrossLoading}
        error={saberCrossError}
        onRetry={runQuery}
        topK={topK}
        onInspectCandidate={c => setInspectorCandidate(c)}
      />

      {/* Multi-Sensor Inspector Modal */}
      <MultiSensorInspector
        open={!!inspectorCandidate}
        onClose={() => setInspectorCandidate(null)}
        query={queryInfo}
        candidate={inspectorCandidate}
      />

    </div>
  )
}
