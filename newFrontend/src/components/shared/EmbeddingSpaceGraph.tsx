'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import {
  Activity,
  Eye,
  Layers,
  Maximize2,
  RefreshCw,
  ZoomIn,
  ZoomOut,
  Sparkles,
  Image as ImageIcon,
  MapPin,
  Tag,
  Database,
  ArrowRight,
} from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

export interface EmbeddingPoint {
  id: number
  name: string
  class_index: number
  dominant_class: string
  color: string
  s2_x: number
  s2_y: number
  s1_x: number
  s1_y: number
  bridged_x: number
  bridged_y: number
  thumbnail?: string
}

export interface ClassLegendItem {
  name: string
  color: string
  class_index: number
}

export interface ApiResponse {
  total_samples: number
  points: EmbeddingPoint[]
  class_legend: ClassLegendItem[]
  manifold_dim: number
  projection_method: string
}

interface EmbeddingSpaceGraphProps {
  maxSamples?: number
}

export default function EmbeddingSpaceGraph({ maxSamples = 350 }: EmbeddingSpaceGraphProps) {
  const [data, setData] = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Interactive state
  const [modality, setModality] = useState<'s2' | 's1' | 'bridged'>('s2')
  const [selectedClass, setSelectedClass] = useState<number | null>(null)
  const [hoveredPoint, setHoveredPoint] = useState<EmbeddingPoint | null>(null)
  const [selectedPoint, setSelectedPoint] = useState<EmbeddingPoint | null>(null)

  // Zoom & Pan
  const [transform, setTransform] = useState({ zoom: 1.0, panX: 0, panY: 0 })
  const isDragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0 })

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  // Fetch data
  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`/api/embedding/points?max_samples=${maxSamples}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json: ApiResponse = await res.json()
      setData(json)
      if (json.points && json.points.length > 0) {
        setSelectedPoint(json.points[0])
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch embedding points')
    } finally {
      setLoading(false)
    }
  }, [maxSamples])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Active points based on modality
  const getPointCoords = useCallback((p: EmbeddingPoint, mod: 's2' | 's1' | 'bridged') => {
    if (mod === 's1') return { x: p.s1_x, y: p.s1_y }
    if (mod === 'bridged') return { x: p.bridged_x, y: p.bridged_y }
    return { x: p.s2_x, y: p.s2_y }
  }, [])

  // Canvas drawing
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data || !data.points) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)

    // Draw background grid lines
    ctx.save()
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)'
    ctx.lineWidth = 1

    const gridSize = 40 * transform.zoom
    const offsetX = (width / 2 + transform.panX) % gridSize
    const offsetY = (height / 2 + transform.panY) % gridSize

    for (let x = offsetX; x < width; x += gridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }
    for (let y = offsetY; y < height; y += gridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Origin axes
    const centerX = width / 2 + transform.panX
    const centerY = height / 2 + transform.panY

    ctx.strokeStyle = 'rgba(251, 186, 114, 0.25)'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(centerX, 0)
    ctx.lineTo(centerX, height)
    ctx.moveTo(0, centerY)
    ctx.lineTo(width, centerY)
    ctx.stroke()
    ctx.restore()

    // Map 2D domain [-12, 12] to Canvas screen coordinates
    const scale = (Math.min(width, height) / 26) * transform.zoom

    const toScreen = (x: number, y: number) => ({
      sx: centerX + x * scale,
      sy: centerY - y * scale,
    })

    // Highlight lines to nearest neighbors if a point is selected
    if (selectedPoint) {
      const selCoords = getPointCoords(selectedPoint, modality)
      const selScreen = toScreen(selCoords.x, selCoords.y)

      // Calculate distances to find 5 nearest neighbors
      const neighbors = data.points
        .filter(p => p.id !== selectedPoint.id)
        .map(p => {
          const c = getPointCoords(p, modality)
          const dist = Math.hypot(c.x - selCoords.x, c.y - selCoords.y)
          return { p, c, dist }
        })
        .sort((a, b) => a.dist - b.dist)
        .slice(0, 5)

      neighbors.forEach(({ c }) => {
        const nScreen = toScreen(c.x, c.y)
        ctx.save()
        ctx.strokeStyle = 'rgba(251, 186, 114, 0.5)'
        ctx.lineWidth = 1.5
        ctx.setLineDash([4, 4])
        ctx.beginPath()
        ctx.moveTo(selScreen.sx, selScreen.sy)
        ctx.lineTo(nScreen.sx, nScreen.sy)
        ctx.stroke()
        ctx.restore()
      })
    }

    // Draw points
    data.points.forEach(point => {
      const isFilteredOut = selectedClass !== null && point.class_index !== selectedClass
      const coords = getPointCoords(point, modality)
      const { sx, sy } = toScreen(coords.x, coords.y)

      if (sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) return

      const isHovered = hoveredPoint?.id === point.id
      const isSelected = selectedPoint?.id === point.id

      ctx.save()
      ctx.globalAlpha = isFilteredOut ? 0.12 : 0.85

      const radius = (isSelected ? 9 : isHovered ? 7.5 : 5) * Math.sqrt(transform.zoom)

      // Outer glow for hovered/selected
      if (isSelected || isHovered) {
        ctx.beginPath()
        ctx.arc(sx, sy, radius + 6, 0, Math.PI * 2)
        ctx.fillStyle = point.color
        ctx.globalAlpha = isFilteredOut ? 0.04 : 0.35
        ctx.fill()
      }

      // Main Circle
      ctx.beginPath()
      ctx.arc(sx, sy, radius, 0, Math.PI * 2)
      ctx.fillStyle = point.color
      ctx.fill()

      // Stroke
      ctx.lineWidth = isSelected ? 2.5 : isHovered ? 2 : 1
      ctx.strokeStyle = isSelected ? '#ffffff' : isHovered ? '#ffffff' : 'rgba(0,0,0,0.5)'
      ctx.stroke()

      ctx.restore()
    })
  }, [data, modality, transform, selectedClass, hoveredPoint, selectedPoint, getPointCoords])

  // Resize canvas to match display size
  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const handleResize = () => {
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Mouse interaction handlers for Pan & Hover
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDragging.current = true
    dragStart.current = { x: e.clientX - transform.panX, y: e.clientY - transform.panY }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || !data || !data.points) return

    if (isDragging.current) {
      setTransform(prev => ({
        ...prev,
        panX: e.clientX - dragStart.current.x,
        panY: e.clientY - dragStart.current.y,
      }))
      return
    }

    const rect = canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2 + transform.panX
    const centerY = height / 2 + transform.panY
    const scale = (Math.min(width, height) / 26) * transform.zoom

    let found: EmbeddingPoint | null = null

    for (const point of data.points) {
      if (selectedClass !== null && point.class_index !== selectedClass) continue
      const coords = getPointCoords(point, modality)
      const sx = centerX + coords.x * scale
      const sy = centerY - coords.y * scale

      const dist = Math.hypot(mouseX - sx, mouseY - sy)
      if (dist <= 12 * Math.sqrt(transform.zoom)) {
        found = point
        break
      }
    }

    setHoveredPoint(found)
  }

  const handleMouseUp = () => {
    isDragging.current = false
  }

  const handleClick = () => {
    if (hoveredPoint) {
      setSelectedPoint(hoveredPoint)
    }
  }

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9
    setTransform(prev => ({
      ...prev,
      zoom: Math.min(Math.max(prev.zoom * zoomFactor, 0.4), 4.0),
    }))
  }

  const handleZoom = (delta: number) => {
    setTransform(prev => ({
      ...prev,
      zoom: Math.min(Math.max(prev.zoom + delta, 0.4), 4.0),
    }))
  }

  const resetView = () => {
    setTransform({ zoom: 1.0, panX: 0, panY: 0 })
    setSelectedClass(null)
  }

  const activePoint = hoveredPoint || selectedPoint

  return (
    <div className="w-full flex flex-col gap-5 font-sans">
      {/* Header controls & stats */}
      <Card className="border-border/60 shadow-sm overflow-hidden bg-card/60 backdrop-blur-md">
        <CardContent className="flex flex-wrap items-center justify-between gap-4 p-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-[#FBBA72]/10 border border-[#FBBA72]/30 text-[#FBBA72]">
              <Database className="size-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-foreground flex items-center gap-2 font-sans">
                2D Embedding Manifold
                <Badge variant="outline" className="border-[#FBBA72]/40 text-[#FBBA72] bg-[#FBBA72]/10 text-[10px] font-mono">
                  768D → 2D Projection
                </Badge>
              </h2>
              <p className="text-xs text-muted-foreground font-sans">
                {data ? `Projected ${data.total_samples} image embeddings from saber_search_db.pth` : 'Loading embeddings...'}
              </p>
            </div>
          </div>

          {/* Modality selector buttons */}
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-background/80 border border-border/60">
            <button
              onClick={() => setModality('s2')}
              className={cn(
                'px-3 py-1.5 text-xs font-semibold rounded-lg transition-all font-sans',
                modality === 's2'
                  ? 'bg-sky-500/15 border border-sky-500/40 text-sky-400 shadow-xs'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/40',
              )}
            >
              Optical (Sentinel-2)
            </button>
            <button
              onClick={() => setModality('s1')}
              className={cn(
                'px-3 py-1.5 text-xs font-semibold rounded-lg transition-all font-sans',
                modality === 's1'
                  ? 'bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 shadow-xs'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/40',
              )}
            >
              SAR (Sentinel-1)
            </button>
            <button
              onClick={() => setModality('bridged')}
              className={cn(
                'px-3 py-1.5 text-xs font-semibold rounded-lg transition-all font-sans',
                modality === 'bridged'
                  ? 'bg-[#FBBA72]/20 border border-[#FBBA72]/50 text-[#FBBA72] shadow-xs'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/40',
              )}
            >
              SABER Bridged (CFM)
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Main Canvas & Inspector Viewport */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-5">
        {/* Interactive Canvas Graph (3 Columns) */}
        <div
          ref={containerRef}
          className="lg:col-span-3 relative h-[580px] w-full rounded-2xl border border-border/60 bg-card/90 overflow-hidden select-none cursor-grab active:cursor-grabbing shadow-sm"
        >
          {loading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-card/80 backdrop-blur-md z-20">
              <RefreshCw className="size-8 text-[#FBBA72] animate-spin mb-3" />
              <span className="text-sm font-medium text-foreground font-sans">Computing 2D PCA Projections...</span>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-card/90 z-20 p-4">
              <div className="p-4 rounded-xl border border-destructive/30 bg-destructive/10 text-destructive text-center max-w-sm">
                <p className="font-semibold text-sm font-sans">Failed to load graph</p>
                <p className="text-xs mt-1">{error}</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={fetchData}
                  className="mt-3 text-xs"
                >
                  Retry
                </Button>
              </div>
            </div>
          )}

          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleClick}
            onWheel={handleWheel}
            className="w-full h-full block"
          />

          {/* Floating Zoom/Pan Controls */}
          <div className="absolute bottom-4 left-4 flex items-center gap-1 p-1 rounded-xl bg-background/80 border border-border/60 backdrop-blur-md z-10 shadow-sm">
            <button
              onClick={() => handleZoom(0.2)}
              className="p-1.5 hover:bg-muted rounded-lg text-muted-foreground hover:text-foreground transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="size-4" />
            </button>
            <button
              onClick={() => handleZoom(-0.2)}
              className="p-1.5 hover:bg-muted rounded-lg text-muted-foreground hover:text-foreground transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="size-4" />
            </button>
            <button
              onClick={resetView}
              className="p-1.5 hover:bg-muted rounded-lg text-muted-foreground hover:text-foreground transition-colors"
              title="Reset View"
            >
              <Maximize2 className="size-4" />
            </button>
          </div>

          {/* Active Modality Indicator Badge */}
          <div className="absolute top-4 left-4 px-3 py-1 rounded-full border border-border/60 bg-background/80 backdrop-blur-md text-xs font-mono text-foreground flex items-center gap-2 z-10 shadow-sm">
            <span
              className={cn(
                'size-2 rounded-full animate-pulse',
                modality === 's2' ? 'bg-sky-400' : modality === 's1' ? 'bg-emerald-400' : 'bg-[#FBBA72]',
              )}
            />
            <span>
              View: {modality === 's2' ? 'Optical (S2)' : modality === 's1' ? 'SAR (S1)' : 'SABER Bridged (CFM)'}
            </span>
          </div>
        </div>

        {/* Selected / Hovered Sample Inspector Panel */}
        <div className="lg:col-span-1 flex flex-col gap-4">
          <Card className="border-border/60 bg-card/80 backdrop-blur-md shadow-sm h-full flex flex-col">
            <CardContent className="p-4 flex flex-col gap-3.5 h-full">
              <div className="flex items-center justify-between border-b border-border/50 pb-3">
                <h3 className="text-sm font-bold text-foreground flex items-center gap-2 font-sans">
                  <Eye className="size-4 text-[#FBBA72]" />
                  Scene Inspector
                </h3>
                <span className="text-xs font-mono text-muted-foreground">
                  {activePoint ? `#${activePoint.id}` : 'Select a node'}
                </span>
              </div>

              {activePoint ? (
                <div className="flex flex-col gap-3.5 flex-1 justify-between">
                  <div className="space-y-3">
                    {/* Satellite Thumbnail Preview */}
                    <div className="relative w-full h-48 rounded-xl border border-border/60 bg-muted/20 overflow-hidden group">
                      {activePoint.thumbnail ? (
                        <img
                          src={activePoint.thumbnail}
                          alt={activePoint.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground/40">
                          <ImageIcon className="size-8 mb-1" />
                          <span className="text-xs font-sans">Thumbnail unavailable</span>
                        </div>
                      )}

                      <div className="absolute bottom-2 left-2 right-2 px-2.5 py-1 rounded-lg bg-background/80 border border-border/60 backdrop-blur-md flex items-center justify-between text-xs">
                        <span className="font-mono text-foreground truncate max-w-[140px]" title={activePoint.name}>
                          {activePoint.name}
                        </span>
                        <span className="text-[10px] text-[#FBBA72] font-mono">120x120</span>
                      </div>
                    </div>

                    {/* Class Label Badge */}
                    <div className="flex items-center justify-between text-xs p-2.5 rounded-xl bg-muted/20 border border-border/40 font-sans">
                      <span className="text-muted-foreground flex items-center gap-1.5">
                        <Tag className="size-3.5" /> Class
                      </span>
                      <span
                        className="font-semibold px-2.5 py-0.5 rounded-md text-foreground"
                        style={{ backgroundColor: `${activePoint.color}25`, border: `1px solid ${activePoint.color}` }}
                      >
                        {activePoint.dominant_class}
                      </span>
                    </div>

                    {/* 2D Coordinates Detail */}
                    <div className="space-y-2 p-3 rounded-xl bg-muted/20 border border-border/40 text-xs font-mono">
                      <div className="flex justify-between text-muted-foreground">
                        <span>Optical (S2):</span>
                        <span className="text-sky-400 font-semibold">({activePoint.s2_x}, {activePoint.s2_y})</span>
                      </div>
                      <div className="flex justify-between text-muted-foreground">
                        <span>SAR (S1):</span>
                        <span className="text-emerald-400 font-semibold">({activePoint.s1_x}, {activePoint.s1_y})</span>
                      </div>
                      <div className="flex justify-between text-muted-foreground">
                        <span>CFM Bridged:</span>
                        <span className="text-[#FBBA72] font-semibold">({activePoint.bridged_x}, {activePoint.bridged_y})</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-64 flex flex-col items-center justify-center text-center p-4 text-muted-foreground my-auto">
                  <MapPin className="size-8 mb-2 opacity-40 text-[#FBBA72]" />
                  <p className="text-xs font-medium text-foreground font-sans">Hover or click any node on the graph</p>
                  <p className="text-[11px] text-muted-foreground mt-1 font-sans">
                    Preview satellite scene details and manifold coordinates.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Land Cover Class Legend Filter */}
      {data && data.class_legend && (
        <Card className="border-border/60 bg-card/60 backdrop-blur-md shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2 font-sans">
                <Layers className="size-3.5 text-[#FBBA72]" />
                Land Cover Clusters ({data.class_legend.length} Classes)
              </h4>
              {selectedClass !== null && (
                <button
                  onClick={() => setSelectedClass(null)}
                  className="text-xs text-[#FBBA72] hover:underline font-medium font-sans"
                >
                  Clear Filter
                </button>
              )}
            </div>

            <div className="flex flex-wrap gap-2">
              {data.class_legend.map(cls => {
                const isSelected = selectedClass === cls.class_index
                return (
                  <button
                    key={cls.class_index}
                    onClick={() => setSelectedClass(isSelected ? null : cls.class_index)}
                    className={cn(
                      'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-all font-sans border',
                      isSelected
                        ? 'border-[#FBBA72] bg-[#FBBA72]/15 text-foreground font-semibold shadow-xs'
                        : 'border-border/40 bg-muted/20 text-muted-foreground hover:text-foreground hover:bg-muted/40',
                    )}
                  >
                    <span className="size-2.5 rounded-full" style={{ backgroundColor: cls.color }} />
                    <span>{cls.name}</span>
                  </button>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
