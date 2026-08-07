import React from 'react'
import EmbeddingSpaceGraph from '@/components/shared/EmbeddingSpaceGraph'

export const metadata = {
  title: 'Embedding Space Graph | SABER Platform',
  description: 'Interactive 2D manifold projection of image embeddings from saber_search_db.pth',
}

export default function EmbeddingSpacePage() {
  return (
    <div className="w-full max-w-7xl mx-auto p-4 md:p-6 space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
          Embedding Space Graph
          <span className="text-xs px-2.5 py-1 rounded-full border border-purple-500/30 bg-purple-500/10 text-purple-300 font-mono">
            Zero-GPU Search DB
          </span>
        </h1>
        <p className="text-sm text-neutral-400">
          Explore high-dimensional remote sensing image embeddings mapped onto a 2D cosine-preserving manifold.
          Similar images are clustered close together, while dissimilar images are positioned farther apart.
        </p>
      </div>

      {/* Main Graph Component */}
      <EmbeddingSpaceGraph maxSamples={350} />
    </div>
  )
}
