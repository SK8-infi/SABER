'use client'

import { useEffect, useState, useCallback } from 'react'
import { AlertCircle, RotateCw, ArrowRight, ShieldCheck, XCircle, EyeIcon } from 'lucide-react'

import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useRetrievalParams } from '@/contexts/retrieval-params-context'
import MultiSensorInspector from '@/components/shared/MultiSensorInspector'

/* ── API types ─────────────────────────────────────────────── */
interface AblationCandidate {
  rank: number
  name: string
  thumbnail?: string
  similarity_score: number
  jaccard_overlap: number
  active_classes?: string[]
}

interface AblationSide {
  avg_similarity: number
  avg_jaccard: number
  candidates: AblationCandidate[]
}

interface AblationDelta {
  f1_at_5_baseline: number
  f1_at_5_saber: number
  map_baseline: number
  map_saber: number
  similarity_improvement: number
  jaccard_improvement: number
}

interface AblationResult {
  delta: AblationDelta
  bridge_off: AblationSide
  bridge_on: AblationSide
  query?: {
    name: string
    source_modality: string
    active_classes: string[]
    thumbnail: string
  }
}

/* ── similarity bar ────────────────────────────────────────── */
function SimBar({ pct, color }: { pct: number; color: string }) {
  const clamped = Math.min(Math.max(pct, 0), 100)
  return (
    <div className='w-full h-1.5 bg-muted/60 rounded-full overflow-hidden'>
      <div className='h-full rounded-full transition-all duration-300' style={{ width: `${clamped}%`, backgroundColor: color }} />
    </div>
  )
}

/* ── candidate row ─────────────────────────────────────────── */
function CandidateRow({ c, color, onInspect }: { c: AblationCandidate; color: string; onInspect?: () => void }) {
  return (
    <div className='flex items-center gap-3 py-2.5 border-b border-border/30 last:border-0'>
      <span className='text-xs font-mono text-muted-foreground w-7 shrink-0'>#{c.rank}</span>
      {c.thumbnail && (
        <img src={c.thumbnail} alt={c.name} className='size-10 rounded-lg object-cover shrink-0 border border-border/40' loading='lazy' />
      )}
      <span className='text-sm text-foreground font-sans truncate flex-1'>{c.name}</span>
      <div className='w-20 shrink-0 hidden sm:block'>
        <SimBar pct={c.similarity_score} color={color} />
      </div>
      <span className='text-sm font-mono font-bold shrink-0' style={{ color }}>{c.similarity_score}%</span>
      {onInspect && (
        <Button
          variant='ghost'
          size='sm'
          onClick={onInspect}
          className='size-7 p-0 shrink-0 text-muted-foreground hover:text-foreground hover:bg-muted/50'
          title='Inspect'
        >
          <EyeIcon className='size-3.5' />
        </Button>
      )}
    </div>
  )
}

/* ── skeleton row ──────────────────────────────────────────── */
function SkeletonRow() {
  return (
    <div className='flex items-center gap-3 py-2.5 border-b border-border/30 last:border-0 animate-pulse'>
      <div className='w-7 h-3.5 rounded bg-muted/40' />
      <div className='size-10 rounded-lg bg-muted/40 shrink-0' />
      <div className='flex-1 h-3.5 rounded bg-muted/40' />
      <div className='w-20 h-1.5 rounded bg-muted/40' />
      <div className='w-10 h-3.5 rounded bg-muted/40' />
    </div>
  )
}

/* ── delta badge ───────────────────────────────────────────── */
function DeltaBadge({ label, base, saber }: { label: string; base: number; saber: number }) {
  const diff = saber - base
  const positive = diff >= 0
  return (
    <div className='flex flex-col items-center gap-0.5 px-4 py-2 rounded-lg bg-muted/20 border border-border/40'>
      <span className='text-xs font-semibold uppercase tracking-wider text-muted-foreground font-sans'>{label}</span>
      <div className='flex items-center gap-1.5 text-sm font-mono'>
        <span className='text-[#FBBA72]'>{base}</span>
        <ArrowRight className='size-3.5 text-muted-foreground' />
        <span className='font-bold text-[#FBBA72]'>{saber}</span>
      </div>
      <span className='text-xs font-bold font-mono text-[#FBBA72]'>
        {positive ? '+' : ''}{diff.toFixed ? diff.toFixed(2) : diff}
      </span>
    </div>
  )
}

/* ── main page ─────────────────────────────────────────────── */
export default function AblationPage() {
  const { params, setTelemetry } = useRetrievalParams()
  const { dataset, qIdx } = params

  const isBen  = dataset === 'ben14k'
  const srcMod = isBen ? 's1'  : 'pan'
  const tgtMod = isBen ? 's2'  : 'ms'

  const [result,  setResult]  = useState<AblationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [inspectorCandidate, setInspectorCandidate] = useState<AblationCandidate | null>(null)

  const run = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/retrieval/ablation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name:    dataset,
          query_index:     qIdx,
          source_modality: srcMod,
          target_modality: tgtMod,
          top_k: 5,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setResult(data)
      if (data?.bridge_on?.telemetry?.total_latency_ms) {
        setTelemetry({ total_latency_ms: data.bridge_on.telemetry.total_latency_ms })
      }
    } catch (err: any) {
      setError(err?.message ?? 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [dataset, qIdx, srcMod, tgtMod, setTelemetry])

  useEffect(() => { run() }, [run])

  return (
    <div className='w-full space-y-8'>

      {/* Error state */}
      {error && !loading && (
        <Card className='border-rose-500/20 bg-rose-500/5'>
          <CardContent className='p-4 flex items-center justify-between gap-3'>
            <div className='flex items-center gap-2 text-sm text-rose-400 font-sans'>
              <AlertCircle className='size-4 shrink-0' />
              <span>{error}</span>
            </div>
            <Button variant='outline' size='sm' onClick={run} className='gap-2 font-sans shrink-0'>
              <RotateCw className='size-3.5' /> Retry
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Delta summary */}
      {(result || loading) && (
        <Card className='overflow-hidden border-border/60 border-t-4 border-t-[#FBBA72] py-0'>
          <CardContent className='p-5 space-y-5'>
            {/* Header row */}
            <div className='flex items-center justify-between flex-wrap gap-2'>
              <div className='flex flex-col gap-0.5'>
                <h3 className='text-base font-bold text-foreground font-sans'>Ablation Study</h3>
                <p className='text-xs text-muted-foreground font-sans'>Comparing Bridge OFF (baseline) vs Bridge ON (SABER)</p>
              </div>
              <div className='flex items-center gap-2'>
                <span className='text-xs font-sans text-muted-foreground bg-muted/50 border border-border/50 px-2.5 py-1 rounded-full'>Bridge OFF</span>
                <ArrowRight className='size-3.5 text-[#FBBA72]' />
                <span className='text-xs font-sans text-[#FBBA72] bg-[#FBBA72]/10 border border-[#FBBA72]/30 px-2.5 py-1 rounded-full font-semibold'>Bridge ON</span>
              </div>
            </div>
            {/* Stats grid */}
            <div className='grid grid-cols-2 sm:grid-cols-4 gap-3'>
              {loading ? (
                Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className='h-20 rounded-xl bg-muted/30 animate-pulse' />
                ))
              ) : result ? (
                <>
                  {[
                    { label: 'F1@5', base: result.delta.f1_at_5_baseline, saber: result.delta.f1_at_5_saber },
                    { label: 'mAP',  base: result.delta.map_baseline,     saber: result.delta.map_saber },
                  ].map((m) => {
                    const numBase = parseFloat(String(m.base))
                    const numSaber = parseFloat(String(m.saber))
                    const diff = !isNaN(numBase) && !isNaN(numSaber) ? numSaber - numBase : null
                    return (
                      <div key={m.label} className='flex flex-col gap-2 rounded-xl bg-muted/20 border border-border/40 p-3.5'>
                        <span className='text-xs font-semibold uppercase tracking-wider text-muted-foreground font-sans'>{m.label}</span>
                        <div className='flex items-end gap-2'>
                          <span className='text-xl font-bold text-[#FBBA72] font-sans'>{m.saber}</span>
                          <span className='text-xs text-muted-foreground font-sans mb-0.5 line-through'>{m.base}</span>
                        </div>
                        {diff !== null && (
                          <span className={`text-xs font-semibold font-sans ${diff >= 0 ? 'text-[#FBBA72]' : 'text-rose-400'}`}>
                            {diff >= 0 ? '↑' : '↓'} +{Math.abs(diff).toFixed(2)} pp gain
                          </span>
                        )}
                      </div>
                    )
                  })}
                  <div className='flex flex-col gap-2 rounded-xl bg-[#FBBA72]/5 border border-[#FBBA72]/20 p-3.5'>
                    <span className='text-xs font-semibold uppercase tracking-wider text-muted-foreground font-sans'>Sim Δ</span>
                    <span className='text-2xl font-bold text-[#FBBA72] font-sans'>+{result.delta.similarity_improvement}%</span>
                    <span className='text-xs text-muted-foreground font-sans'>similarity gain</span>
                  </div>
                  <div className='flex flex-col gap-2 rounded-xl bg-[#FBBA72]/5 border border-[#FBBA72]/20 p-3.5'>
                    <span className='text-xs font-semibold uppercase tracking-wider text-muted-foreground font-sans'>Jaccard Δ</span>
                    <span className='text-2xl font-bold text-[#FBBA72] font-sans'>+{result.delta.jaccard_improvement}%</span>
                    <span className='text-xs text-muted-foreground font-sans'>overlap gain</span>
                  </div>
                </>
              ) : null}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Side-by-side tables */}
      <div className='grid grid-cols-1 lg:grid-cols-2 gap-6'>

        {/* Model 1 — Bridge OFF (Baseline) */}
        <Card>
          <CardContent className='p-4 space-y-3'>
            <div className='flex items-center justify-between'>
              <div className='flex items-center gap-2'>
                <XCircle className='size-4 text-[#FBBA72]' />
                <h2 className='text-base font-semibold font-sans text-foreground'>Model 1 — Baseline</h2>
                <span className='text-xs font-sans text-muted-foreground bg-muted/40 px-2 py-0.5 rounded'>Bridge OFF</span>
              </div>
              {result && (
                <div className='flex items-center gap-2 text-xs font-mono text-muted-foreground'>
                  <span>Sim <span className='text-foreground font-bold'>{result.bridge_off.avg_similarity}%</span></span>
                  <span>Jac <span className='text-foreground font-bold'>{result.bridge_off.avg_jaccard}%</span></span>
                </div>
              )}
            </div>

            <div className='divide-y divide-border/30'>
              {loading
                ? Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)
                : result?.bridge_off.candidates.map(c => (
                    <CandidateRow key={c.rank} c={c} color='#f87171' onInspect={() => setInspectorCandidate(c)} />
                  ))}
            </div>
          </CardContent>
        </Card>

        {/* Model 2 — Bridge ON (SABER) */}
        <Card>
          <CardContent className='p-4 space-y-3'>
            <div className='flex items-center justify-between'>
              <div className='flex items-center gap-2'>
                <ShieldCheck className='size-4 text-[#FBBA72]' />
                <h2 className='text-base font-semibold font-sans text-foreground'>Model 2 — SABER</h2>
                <span className='text-xs font-sans text-muted-foreground bg-muted/40 px-2 py-0.5 rounded'>Bridge ON</span>
              </div>
              {result && (
                <div className='flex items-center gap-2 text-xs font-mono text-muted-foreground'>
                  <span>Sim <span className='text-[#FBBA72] font-bold'>{result.bridge_on.avg_similarity}%</span></span>
                  <span>Jac <span className='text-[#FBBA72] font-bold'>{result.bridge_on.avg_jaccard}%</span></span>
                </div>
              )}
            </div>

            <div className='divide-y divide-border/30'>
              {loading
                ? Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)
                : result?.bridge_on.candidates.map(c => (
                    <CandidateRow key={c.rank} c={c} color='#34d399' onInspect={() => setInspectorCandidate(c)} />
                  ))}
            </div>
          </CardContent>
        </Card>

      </div>

      {/* Multi-Sensor Inspector Modal */}
      <MultiSensorInspector
        open={!!inspectorCandidate}
        onClose={() => setInspectorCandidate(null)}
        query={result?.query ?? {
          name: `Scene #${qIdx}`,
          source_modality: srcMod,
          active_classes: [],
          thumbnail: inspectorCandidate?.thumbnail ?? ''
        }}
        candidate={inspectorCandidate ? {
          name: inspectorCandidate.name,
          rank: inspectorCandidate.rank,
          similarity_score: inspectorCandidate.similarity_score,
          jaccard_overlap: inspectorCandidate.jaccard_overlap,
          active_classes: inspectorCandidate.active_classes ?? [],
          thumbnail: inspectorCandidate.thumbnail ?? ''
        } : null}
      />

    </div>
  )
}
