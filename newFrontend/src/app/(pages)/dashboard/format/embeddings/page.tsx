import React from 'react'
import EmbeddingSpaceGraph from '@/components/shared/EmbeddingSpaceGraph'
import { Badge } from '@/components/ui/badge'
import { SparklesIcon } from 'lucide-react'

export const metadata = {
  title: 'Embedding Space Graph | SABER Platform',
  description: 'Interactive 2D manifold projection of image embeddings from saber_search_db.pth',
}

export default function EmbeddingSpacePage() {
  return (
    <div className="w-full space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-2 border-b border-border/40 pb-5">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-3 font-sans">
            Embedding Space Graph
          </h1>
          <Badge
            variant="outline"
            className="border-[#FBBA72]/40 text-[#FBBA72] bg-[#FBBA72]/10 text-xs font-semibold px-3 py-0.5 rounded-full font-sans"
          >
            <SparklesIcon className="size-3 mr-1 text-[#FBBA72]" />
            768D → 2D Manifold
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground max-w-3xl font-sans leading-relaxed">
          Explore high-dimensional satellite image embeddings mapped onto a 2D metric-preserving manifold.
          Optical (Sentinel-2) and SAR (Sentinel-1) embeddings are unified in real-time via Continuous Flow Matching (CFM) ODE latent translation.
        </p>
      </div>

      {/* Main Graph Component */}
      <EmbeddingSpaceGraph maxSamples={350} />
    </div>
  )
}
