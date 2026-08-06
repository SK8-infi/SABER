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
  const { dataset, srcMod, tgtMod, qIdx, topK, bridge, rerank, odeSteps } = params

  const [saberResult, setSaberResult] = useState<QueryResult | null>(null)
  const [isroResult,  setIsroResult]  = useState<QueryResult | null>(null)
  const [saberLoading, setSaberLoading] = useState(false)
  const [isroLoading,  setIsroLoading]  = useState(false)
  const [saberError, setSaberError] = useState<string | null>(null)
  const [isroError,  setIsroError]  = useState<string | null>(null)

  const runQuery = useCallback(async () => {
    setSaberLoading(true)
    setIsroLoading(true)
    setSaberError(null)
    setIsroError(null)

    const body = (enable_bridge: boolean, enable_rerank: boolean) => ({
      dataset_name:    dataset,
      query_index:     qIdx,
      source_modality: srcMod,
      target_modality: tgtMod,
      top_k:           topK,
      enable_bridge,
      enable_rerank,
      ode_steps:       odeSteps,
    })

    // fire both in parallel — SABER (bridge ON) and ISRO baseline (bridge OFF)
    const [saberRes, isroRes] = await Promise.allSettled([
      fetch('/api/retrieval/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body(bridge, rerank)),
      }),
      fetch('/api/retrieval/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body(false, false)),
      }),
    ])

    // SABER
    if (saberRes.status === 'fulfilled') {
      if (saberRes.value.ok) {
        const data = await saberRes.value.json()
        setSaberResult(data)
        if (data?.latency_telemetry?.total_latency_ms) {
          setTelemetry({ total_latency_ms: data.latency_telemetry.total_latency_ms })
        }
      } else {
        setSaberError(`SABER: HTTP ${saberRes.value.status}`)
      }
    } else {
      setSaberError(`SABER: ${saberRes.reason}`)
    }
    setSaberLoading(false)

    // ISRO
    if (isroRes.status === 'fulfilled') {
      if (isroRes.value.ok) {
        setIsroResult(await isroRes.value.json())
      } else {
        setIsroError(`ISRO: HTTP ${isroRes.value.status}`)
      }
    } else {
      setIsroError(`ISRO: ${isroRes.reason}`)
    }
    setIsroLoading(false)
  }, [dataset, qIdx, srcMod, tgtMod, topK, bridge, rerank, odeSteps, setTelemetry])

  // re-run whenever any param changes
  useEffect(() => { runQuery() }, [runQuery])

  // resolve query info for the top card (prefer SABER, fallback ISRO)
  const queryInfo = saberResult?.query ?? isroResult?.query
  const topLoading = saberLoading && isroLoading

  const [inspectorCandidate, setInspectorCandidate] = useState<Candidate | null>(null)

  return (
    <div className='w-full space-y-6'>

      {/* ── Top query image card ── */}
      <Card className='border-border/60 shadow-sm overflow-hidden'>
        <CardContent className='p-0'>
          <div className='flex flex-col sm:flex-row'>

            {/* Image panel */}
            <div className='relative w-full sm:w-48 aspect-square sm:shrink-0 bg-muted/20 overflow-hidden self-center'>
              {topLoading ? (
                <div className='w-full h-full bg-muted/30 animate-pulse' />
              ) : queryInfo?.thumbnail ? (
                <img src={queryInfo.thumbnail} alt={queryInfo.name} className='w-full h-full object-cover' />
              ) : (
                <div className='w-full h-full bg-muted/20 flex items-center justify-center'>
                  <SearchIcon className='size-8 text-muted-foreground/30' />
                </div>
              )}
              <span className='absolute top-2 left-2 bg-black/75 backdrop-blur-sm text-white text-[10px] font-semibold px-2 py-0.5 rounded-full font-sans tracking-wider border border-white/10'>
                {queryInfo?.source_modality?.toUpperCase() ?? srcMod.toUpperCase()}
              </span>
            </div>

            {/* Info panel */}
            <div className='flex flex-col justify-between gap-4 p-5 flex-1 min-w-0'>

              {/* Filename + class badges */}
              <div className='flex flex-col gap-2'>
                <div className='flex items-start justify-between gap-2 flex-wrap'>
                  <h3 className='text-sm font-bold text-foreground font-sans tracking-tight leading-tight'>
                    {topLoading
                      ? <span className='inline-block w-48 h-4 rounded bg-muted/40 animate-pulse' />
                      : (queryInfo?.name ?? `Scene #${qIdx}`)}
                  </h3>
                  {queryInfo && (
                    <div className='flex flex-wrap gap-1'>
                      {queryInfo.active_classes.slice(0, 2).map((cl, i) => (
                        <Badge
                          key={i}
                          variant='outline'
                          className='border-[#FBBA72]/40 text-[#FBBA72] bg-[#FBBA72]/10 text-[10px] font-semibold px-2 py-0.5 rounded-full font-sans shrink-0'
                        >
                          {cl}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
                <p className='text-[11px] text-muted-foreground font-sans'>Query image for cross-modal retrieval</p>
              </div>

              {/* Divider */}
              <div className='h-px bg-border/40 w-full' />

              {/* Meta grid */}
              <div className='grid grid-cols-3 gap-3'>
                <div className='flex flex-col gap-0.5'>
                  <span className='text-[9px] font-semibold uppercase tracking-widest text-muted-foreground font-sans'>Target Gallery</span>
                  <span className='text-sm font-bold text-foreground font-sans'>
                    {queryInfo?.target_modality?.toUpperCase() ?? tgtMod.toUpperCase()}
                  </span>
                </div>
                <div className='flex flex-col gap-0.5'>
                  <span className='text-[9px] font-semibold uppercase tracking-widest text-muted-foreground font-sans'>Retrieve</span>
                  <span className='text-sm font-bold text-foreground font-sans'>Top-{topK}</span>
                </div>
                <div className='flex flex-col gap-0.5'>
                  <span className='text-[9px] font-semibold uppercase tracking-widest text-muted-foreground font-sans'>CFM Bridge</span>
                  <span className={`text-sm font-bold font-sans ${bridge ? 'text-[#FBBA72]' : 'text-muted-foreground'}`}>
                    {bridge ? 'ON' : 'OFF'}
                  </span>
                </div>
              </div>

            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── MODEL 1: SABER ── */}
      <ModelSection
        label='MODEL 1'
        modelName='SABER — Neural ODE Bridge (Ours)'
        logo='/images/brands/logo-square.webp'
        result={saberResult}
        loading={saberLoading}
        error={saberError}
        onRetry={runQuery}
        topK={topK}
        onInspectCandidate={c => setInspectorCandidate(c)}
      />

      {/* ── MODEL 2: ISRO ── */}
      <ModelSection
        label='MODEL 2'
        modelName='ISRO Best Model'
        logo='/images/brands/isro-logo.png'
        logoBg='bg-black'
        result={isroResult}
        loading={isroLoading}
        error={isroError}
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
