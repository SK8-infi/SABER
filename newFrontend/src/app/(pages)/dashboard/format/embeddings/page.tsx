import React from 'react'
import EmbeddingSpaceGraph from '@/components/shared/EmbeddingSpaceGraph'
import CustomImageQueryUpload from '@/components/shared/CustomImageQueryUpload'
import { Badge } from '@/components/ui/badge'
import { SparklesIcon, LayersIcon } from 'lucide-react'

export const metadata = {
  title: 'Interactive Query Workspace | SABER Platform',
  description: 'Primary interactive 2D manifold query engine for satellite image embeddings',
}

export default function EmbeddingSpacePage() {
  return (
    <div className="w-full space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-2 border-b border-border/40 pb-5">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-3 font-sans">
            Interactive Query Workspace
          </h1>
          <Badge
            variant="outline"
            className="border-emerald-500/40 text-emerald-400 bg-emerald-500/10 text-xs font-semibold px-3 py-0.5 rounded-full font-sans"
          >
            <SparklesIcon className="size-3 mr-1 text-emerald-400" />
            DEFAULT QUERY ENGINE
          </Badge>
          <Badge
            variant="outline"
            className="border-[#FBBA72]/40 text-[#FBBA72] bg-[#FBBA72]/10 text-xs font-semibold px-3 py-0.5 rounded-full font-sans"
          >
            <LayersIcon className="size-3 mr-1 text-[#FBBA72]" />
            768D → 2D Metric Manifold
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground max-w-3xl font-sans leading-relaxed">
          Primary interactive query dashboard. Explore high-dimensional Sentinel-1 SAR and Sentinel-2 Optical satellite embeddings mapped onto a metric-preserving 2D manifold. Click any node or filter land-cover clusters to trigger real-time Same-Modal and CFM ODE Cross-Modal retrieval queries.
        </p>
      </div>

      {/* Main Graph Component */}
      <EmbeddingSpaceGraph maxSamples={1000} />

      {/* Custom Image Upload Query Engine */}
      <CustomImageQueryUpload />
    </div>
  )
}
