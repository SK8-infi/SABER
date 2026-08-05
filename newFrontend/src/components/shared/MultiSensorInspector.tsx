'use client'

import { XIcon } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useRetrievalParams } from '@/contexts/retrieval-params-context'

export interface InspectorQueryInfo {
  name: string
  source_modality: string
  active_classes: string[]
  thumbnail: string
}

export interface InspectorCandidateInfo {
  name: string
  rank: number
  similarity_score: number
  jaccard_overlap: number
  active_classes: string[]
  thumbnail: string
}

interface MultiSensorInspectorProps {
  open: boolean
  onClose: () => void
  query?: InspectorQueryInfo | null
  candidate?: InspectorCandidateInfo | null
}

export default function MultiSensorInspector({ open, onClose, query, candidate }: MultiSensorInspectorProps) {
  const { telemetry } = useRetrievalParams()

  if (!open || !query || !candidate) return null

  const gallerySize = telemetry.gallery_size ? telemetry.gallery_size.toLocaleString() : '11,866'

  return (
    <div
      className='fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-3 sm:p-4 overflow-hidden animate-in fade-in-0 duration-150'
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <div className='relative w-full max-w-3xl max-h-[95vh] flex flex-col justify-between rounded-2xl border border-border/80 bg-card p-4 sm:p-5 shadow-2xl gap-3 sm:gap-4 text-foreground font-sans overflow-hidden'>

        {/* Modal Header */}
        <div className='flex items-start justify-between border-b border-border/40 pb-3 shrink-0'>
          <div className='flex flex-col gap-0.5'>
            <h2 className='text-sm sm:text-base font-bold uppercase tracking-wider text-foreground font-sans leading-none'>
              MULTI-SENSOR INSPECTOR
            </h2>
            <p className='text-[11px] text-muted-foreground font-sans'>
              Cross-modal pair comparison
            </p>
          </div>
          <Button
            variant='ghost'
            size='icon-sm'
            onClick={onClose}
            className='rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 cursor-pointer shrink-0'
          >
            <XIcon className='size-4' />
            <span className='sr-only'>Close</span>
          </Button>
        </div>

        {/* Image Pair Comparison Grid */}
        <div className='grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4 flex-1 min-h-0'>

          {/* Left Panel: Query Image */}
          <div className='flex flex-col rounded-xl border border-[#FBBA72]/30 bg-muted/10 overflow-hidden flex-1 justify-between min-h-0'>
            <div className='flex items-center justify-between px-3 py-1.5 border-b border-border/40 bg-muted/20 shrink-0'>
              <Badge
                variant='outline'
                className='bg-[#FBBA72]/15 text-[#FBBA72] border-[#FBBA72]/40 text-[10px] font-bold px-2 py-0.5 rounded font-sans uppercase'
              >
                {query.source_modality.toUpperCase()}
              </Badge>
              <span className='text-[10px] font-bold tracking-wider text-muted-foreground uppercase font-sans'>
                QUERY IMAGE
              </span>
            </div>
            <div className='relative w-full h-[26vh] max-h-[220px] bg-black/40 overflow-hidden flex items-center justify-center shrink-0'>
              <img
                src={query.thumbnail}
                alt={query.name}
                className='w-full h-full object-contain bg-black/30'
              />
            </div>
            <div className='p-2.5 flex flex-col gap-1.5 bg-muted/10 border-t border-border/30 shrink-0'>
              <span className='text-xs font-semibold text-foreground truncate font-sans' title={query.name}>
                {query.name}
              </span>
              <div className='flex flex-wrap gap-1 max-h-12 overflow-y-auto no-scrollbar'>
                {query.active_classes.map((cl, i) => (
                  <Badge
                    key={i}
                    variant='secondary'
                    className='bg-muted/80 text-muted-foreground text-[10px] font-medium px-1.5 py-0.5 rounded-md'
                  >
                    {cl}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

          {/* Right Panel: Retrieved Match */}
          <div className='flex flex-col rounded-xl border border-emerald-500/30 bg-muted/10 overflow-hidden flex-1 justify-between min-h-0'>
            <div className='flex items-center justify-between px-3 py-1.5 border-b border-border/40 bg-muted/20 shrink-0'>
              <Badge
                variant='outline'
                className='bg-emerald-500/15 text-emerald-500 border-emerald-500/40 text-[10px] font-bold px-2 py-0.5 rounded font-sans uppercase'
              >
                RANK #{candidate.rank}
              </Badge>
              <span className='text-[10px] font-bold tracking-wider text-muted-foreground uppercase font-sans'>
                RETRIEVED MATCH
              </span>
            </div>
            <div className='relative w-full h-[26vh] max-h-[220px] bg-black/40 overflow-hidden flex items-center justify-center shrink-0'>
              <img
                src={candidate.thumbnail}
                alt={candidate.name}
                className='w-full h-full object-contain bg-black/30'
              />
            </div>
            <div className='p-2.5 flex flex-col gap-1.5 bg-muted/10 border-t border-border/30 shrink-0'>
              <span className='text-xs font-semibold text-foreground truncate font-sans' title={candidate.name}>
                {candidate.name}
              </span>
              <div className='flex flex-wrap gap-1 max-h-12 overflow-y-auto no-scrollbar'>
                {candidate.active_classes.map((cl, i) => (
                  <Badge
                    key={i}
                    variant='secondary'
                    className='bg-muted/80 text-muted-foreground text-[10px] font-medium px-1.5 py-0.5 rounded-md'
                  >
                    {cl}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

        </div>

        {/* Bottom Metrics Bar (3 Columns) */}
        <div className='grid grid-cols-3 divide-x divide-border/40 border border-border/60 rounded-xl bg-muted/20 p-2.5 sm:p-3 shrink-0'>

          {/* Jaccard Overlap */}
          <div className='flex flex-col gap-0.5 px-2 sm:px-4 first:pl-1'>
            <span className='text-[9px] sm:text-[10px] font-bold tracking-wider text-muted-foreground uppercase font-sans truncate'>
              JACCARD OVERLAP
            </span>
            <span className='text-lg sm:text-xl font-mono font-bold text-[#00F0FF] dark:text-[#00F0FF] leading-tight'>
              {candidate.jaccard_overlap}%
            </span>
            <span className='text-[9px] sm:text-[10px] text-muted-foreground font-sans truncate'>
              semantic class similarity
            </span>
          </div>

          {/* Cosine Similarity */}
          <div className='flex flex-col gap-0.5 px-2 sm:px-4'>
            <span className='text-[9px] sm:text-[10px] font-bold tracking-wider text-muted-foreground uppercase font-sans truncate'>
              COSINE SIMILARITY
            </span>
            <span className='text-lg sm:text-xl font-mono font-bold text-[#FBBA72] dark:text-[#FBBA72] leading-tight'>
              {candidate.similarity_score}%
            </span>
            <span className='text-[9px] sm:text-[10px] text-muted-foreground font-sans truncate'>
              embedding space distance
            </span>
          </div>

          {/* Rank Position */}
          <div className='flex flex-col gap-0.5 px-2 sm:px-4 last:pr-1'>
            <span className='text-[9px] sm:text-[10px] font-bold tracking-wider text-muted-foreground uppercase font-sans truncate'>
              RANK POSITION
            </span>
            <span className='text-lg sm:text-xl font-mono font-bold text-emerald-500 dark:text-emerald-400 leading-tight'>
              #{candidate.rank}
            </span>
            <span className='text-[9px] sm:text-[10px] text-muted-foreground font-sans truncate'>
              in gallery of {gallerySize}
            </span>
          </div>

        </div>

      </div>
    </div>
  )
}
