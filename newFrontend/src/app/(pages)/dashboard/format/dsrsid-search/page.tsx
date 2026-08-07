'use client'

import { useState } from 'react'
import {
  Upload,
  Search,
  CheckCircle2,
  AlertCircle,
  Database,
  Zap,
  Layers,
  Image as ImageIcon,
  Sparkles,
  ArrowRight,
  ShieldCheck
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

interface DSRSIDResult {
  rank: number
  similarity: number
  similarity_raw: number
  sample_id: string
  sample_index: number
  class_name: string
  thumbnail: string
  dsrsid_mat_location: string
}

interface SearchResponse {
  status: string
  search_latency_ms: number
  total_gallery_samples: number
  top_k: number
  query_filename: string
  results: DSRSIDResult[]
}

export default function DSRSIDSearchPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null)

  // Handle file select
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setError(null)
      setSearchResponse(null)
    }
  }

  // Execute Search API call to FastAPI backend
  const handleRunSearch = async () => {
    if (!selectedFile) {
      setError('Please select or upload a query image first.')
      return
    }

    setIsSearching(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const apiHost = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiHost}/api/dsrsid/search?top_k=5`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server responded with status ${response.status}`)
      }

      const data: SearchResponse = await response.json()
      setSearchResponse(data)
    } catch (err: any) {
      console.error('DSRSID Search error:', err)
      setError(err.message || 'Failed to execute DSRSID image retrieval search.')
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="flex flex-col gap-6 p-6 max-w-[1600px] mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border/40 pb-5">
        <div>
          <div className="flex items-center gap-2.5 mb-1.5">
            <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/30 px-2.5 py-0.5 text-xs font-semibold">
              ISRO BAH 2026 SOTA MODEL
            </Badge>
            <Badge className="bg-sky-500/10 text-sky-400 border-sky-500/30 px-2.5 py-0.5 text-xs font-semibold">
              1,000 PRE-COMPUTED EMBEDDINGS
            </Badge>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <Database className="w-6 h-6 text-sky-400" />
            DSRSID Gaofen-1 Satellite Image Search
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Upload any query satellite scene (e.g. <span className="font-semibold text-foreground">saber_query.jpeg</span>) to search against <span className="font-semibold text-foreground">1,000 real DSRSID scenes</span> using your trained model checkpoint (<code className="text-xs bg-muted px-1.5 py-0.5 rounded text-sky-300">checkpoints/dsrsid/latest.pth</code>).
          </p>
        </div>

        {/* Telemetry Pill */}
        {searchResponse && (
          <div className="flex items-center gap-3 bg-muted/40 border border-border/60 rounded-xl p-3 text-xs">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <Zap className="w-4 h-4 fill-emerald-400/20" />
              <span>{searchResponse.search_latency_ms} ms Latency</span>
            </div>
            <div className="h-4 w-px bg-border/60" />
            <div className="flex items-center gap-1.5 text-sky-400">
              <Layers className="w-4 h-4" />
              <span>{searchResponse.total_gallery_samples} Gallery Scenes</span>
            </div>
          </div>
        )}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Control Panel (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-5">
          <Card className="border border-border/60 bg-card/60 backdrop-blur-md shadow-xl">
            <CardContent className="p-5 flex flex-col gap-4">
              <h2 className="text-base font-semibold flex items-center gap-2 text-foreground">
                <Upload className="w-4 h-4 text-sky-400" />
                Query Image Input
              </h2>

              {/* Upload Box */}
              <div className="relative border-2 border-dashed border-border/80 hover:border-sky-400/60 rounded-xl p-4 transition-all duration-200 bg-muted/20 hover:bg-muted/30 group text-center flex flex-col items-center justify-center min-h-[220px]">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                />

                {previewUrl ? (
                  <div className="relative w-full aspect-square max-w-[200px] rounded-lg overflow-hidden border border-border/60 shadow-md">
                    <img
                      src={previewUrl}
                      alt="Query Preview"
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-x-0 bottom-0 bg-slate-950/80 backdrop-blur-sm p-1.5 text-[11px] text-sky-300 font-medium truncate text-center">
                      {selectedFile?.name}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2.5">
                    <div className="w-12 h-12 rounded-full bg-sky-500/10 text-sky-400 flex items-center justify-center group-hover:scale-110 transition-transform">
                      <ImageIcon className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-foreground">
                        Drop query satellite scene here
                      </p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Supports JPEG, PNG, JPG (e.g. saber_query.jpeg)
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Run Button */}
              <Button
                onClick={handleRunSearch}
                disabled={!selectedFile || isSearching}
                className="w-full bg-gradient-to-r from-sky-500 to-emerald-500 hover:from-sky-600 hover:to-emerald-600 text-slate-950 font-semibold shadow-lg shadow-sky-500/20 py-5 text-sm"
              >
                {isSearching ? (
                  <span className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                    Searching 1,000 Embeddings...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <Search className="w-4 h-4" />
                    Run 1,000 DSRSID Vector Search
                  </span>
                )}
              </Button>

              {error && (
                <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
                  <p>{error}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Model Specification Card */}
          <Card className="border border-border/60 bg-card/40 backdrop-blur-sm">
            <CardContent className="p-4 flex flex-col gap-3 text-xs">
              <h3 className="font-semibold text-foreground flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                Model Specifications
              </h3>
              <div className="space-y-2 text-muted-foreground">
                <div className="flex justify-between border-b border-border/40 pb-1.5">
                  <span>Checkpoint</span>
                  <span className="font-mono text-sky-300">checkpoints/dsrsid/latest.pth</span>
                </div>
                <div className="flex justify-between border-b border-border/40 pb-1.5">
                  <span>Embedding Space</span>
                  <span className="font-semibold text-foreground">384 Dimensions</span>
                </div>
                <div className="flex justify-between border-b border-border/40 pb-1.5">
                  <span>Dataset Database</span>
                  <span className="font-medium text-emerald-400">1,000 Real DSRSID Scenes</span>
                </div>
                <div className="flex justify-between">
                  <span>Distance Metric</span>
                  <span className="font-medium text-foreground">Cosine Similarity Index</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Search Results Display (8 cols) */}
        <div className="lg:col-span-8 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-400" />
              Top-5 Retrieved Gallery Matches
            </h2>
            {searchResponse && (
              <Badge variant="outline" className="border-emerald-500/30 text-emerald-400 bg-emerald-500/10">
                Found {searchResponse.results.length} Matches
              </Badge>
            )}
          </div>

          {!searchResponse && !isSearching && (
            <Card className="border border-border/40 bg-muted/10 p-12 text-center flex flex-col items-center justify-center gap-3 min-h-[380px]">
              <div className="w-12 h-12 rounded-full bg-muted/40 text-muted-foreground flex items-center justify-center">
                <Search className="w-6 h-6" />
              </div>
              <div>
                <p className="text-base font-medium text-foreground">No Search Executed Yet</p>
                <p className="text-xs text-muted-foreground max-w-md mx-auto mt-1">
                  Upload a query satellite scene on the left and click "Run 1,000 DSRSID Vector Search" to view live retrieved image patches.
                </p>
              </div>
            </Card>
          )}

          {isSearching && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-44 rounded-xl bg-muted/20 border border-border/40 animate-pulse p-4 flex gap-4">
                  <div className="w-32 h-32 rounded-lg bg-muted/40 shrink-0" />
                  <div className="flex flex-col gap-2.5 w-full">
                    <div className="h-4 w-1/3 rounded bg-muted/40" />
                    <div className="h-3 w-1/2 rounded bg-muted/40" />
                    <div className="h-2 w-full rounded bg-muted/40 mt-auto" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {searchResponse && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {searchResponse.results.map((res) => (
                <Card
                  key={res.rank}
                  className={`border transition-all duration-200 hover:shadow-lg ${
                    res.rank === 1
                      ? 'border-emerald-500/50 bg-emerald-500/5 shadow-emerald-500/10'
                      : 'border-border/60 bg-card/60'
                  }`}
                >
                  <CardContent className="p-4 flex gap-4">
                    {/* Thumbnail Image */}
                    <div className="relative w-32 h-32 rounded-lg overflow-hidden border border-border/60 bg-slate-950 shrink-0">
                      <img
                        src={res.thumbnail}
                        alt={res.sample_id}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute top-1.5 left-1.5 px-2 py-0.5 rounded bg-slate-950/80 backdrop-blur-sm text-[10px] font-bold text-emerald-400 border border-emerald-500/30">
                        #{res.rank}
                      </div>
                    </div>

                    {/* Meta & Stats */}
                    <div className="flex flex-col justify-between w-full min-w-0">
                      <div>
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <Badge className="bg-sky-500/10 text-sky-400 border-sky-500/30 text-[11px] font-semibold">
                            {res.class_name.toUpperCase()}
                          </Badge>
                          <span className="text-base font-bold text-emerald-400">
                            {res.similarity}%
                          </span>
                        </div>
                        <p className="text-xs font-medium text-foreground truncate">
                          {res.sample_id}
                        </p>
                        <p className="text-[11px] font-mono text-muted-foreground mt-0.5 truncate">
                          {res.dsrsid_mat_location}
                        </p>
                      </div>

                      {/* Similarity Progress Bar */}
                      <div className="mt-3">
                        <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                          <span>Similarity Score</span>
                          <span>{res.similarity}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-muted/60 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-sky-400 to-emerald-400 rounded-full transition-all duration-500"
                            style={{ width: `${res.similarity}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
