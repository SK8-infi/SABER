'use client'

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'

export interface RetrievalParams {
  dataset: string       // api value e.g. 'ben14k', 'dsrsid'
  srcMod: string        // e.g. 's1', 'pan'
  tgtMod: string        // e.g. 's2', 'ms'
  qIdx: number
  topK: number
  bridge: boolean
  rerank: boolean
  odeSteps: number
}

export interface TelemetryData {
  total_latency_ms: number
  gallery_size: number
  vram_allocated_mb: number
}

interface RetrievalParamsContextValue {
  params: RetrievalParams
  setParams: (patch: Partial<RetrievalParams>) => void
  telemetry: TelemetryData
  setTelemetry: (patch: Partial<TelemetryData>) => void
}

const defaults: RetrievalParams = {
  dataset: 'ben14k',
  srcMod: 's1',
  tgtMod: 's2',
  qIdx: 0,
  topK: 5,
  bridge: true,
  rerank: true,
  odeSteps: 7,
}

const defaultTelemetry: TelemetryData = {
  total_latency_ms: 315.11,
  gallery_size: 1000,
  vram_allocated_mb: 918.7,
}

const RetrievalParamsContext = createContext<RetrievalParamsContextValue>({
  params: defaults,
  setParams: () => {},
  telemetry: defaultTelemetry,
  setTelemetry: () => {},
})

export function RetrievalParamsProvider({ children }: { children: ReactNode }) {
  const [params, setParamsState] = useState<RetrievalParams>(defaults)
  const [telemetry, setTelemetryState] = useState<TelemetryData>(defaultTelemetry)

  const setParams = useCallback((patch: Partial<RetrievalParams>) => {
    setParamsState(prev => ({ ...prev, ...patch }))
  }, [])

  const setTelemetry = useCallback((patch: Partial<TelemetryData>) => {
    setTelemetryState(prev => ({ ...prev, ...patch }))
  }, [])

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(data => {
        if (data) {
          setTelemetryState(prev => ({
            ...prev,
            gallery_size: data.gallery_size ?? prev.gallery_size,
            vram_allocated_mb: data.vram_allocated_mb ?? prev.vram_allocated_mb,
          }))
        }
      })
      .catch(() => {})
  }, [])

  return (
    <RetrievalParamsContext.Provider value={{ params, setParams, telemetry, setTelemetry }}>
      {children}
    </RetrievalParamsContext.Provider>
  )
}

export const useRetrievalParams = () => useContext(RetrievalParamsContext)
