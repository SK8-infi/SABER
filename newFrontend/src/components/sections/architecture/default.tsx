"use client";

import {
  ArrowRightIcon,
  CpuIcon,
  DatabaseIcon,
  GitCommitIcon,
  LayersIcon,
  NetworkIcon,
  RadioIcon,
  ZapIcon,
} from "lucide-react";
import React, { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Glow from "@/components/ui/glow";
import { Section } from "@/components/ui/section";
import { cn } from "@/lib/utils";

interface ArchitectureStep {
  id: string;
  number: string;
  title: string;
  shortDesc: string;
  icon: React.ReactNode;
  tag: string;
  details: {
    overview: string;
    techStack: string[];
    metrics: { label: string; value: string }[];
    formula?: string;
    flowInput: string;
    flowOutput: string;
  };
}

const ARCHITECTURE_STEPS: ArchitectureStep[] = [
  {
    id: "input-ingestion",
    number: "01",
    title: "Multi-Sensor EO Ingestion",
    shortDesc: "Sentinel-1 SAR radar backscatter & Sentinel-2 12-band optical imagery",
    icon: <RadioIcon className="size-5 text-sky-400" />,
    tag: "Sensor Agnostic",
    details: {
      overview:
        "Ingests multi-spectral satellite tiles across optical, infrared, and Synthetic Aperture Radar (SAR) bands. Standardizes spatial resolution and projects raw digital numbers into wavelength-calibrated physical reflectance tensor patches.",
      techStack: ["Sentinel-1 (VV/VH)", "Sentinel-2 (12 Bands)", "Gaofen-1", "DOFA Band Alignment"],
      metrics: [
        { label: "Spatial Res", value: "10m - 20m" },
        { label: "Spectral Range", value: "443nm - 2190nm" },
        { label: "Modality", value: "SAR + Multispectral" },
      ],
      flowInput: "Raw GeoTIFF / Raster Patches",
      flowOutput: "Calibrated Wavelength Tensors (B × C × H × W)",
    },
  },
  {
    id: "vit-encoder",
    number: "02",
    title: "Wavelength ViT & LoRA",
    shortDesc: "Dynamic patch hypernetworks & parameter-efficient LoRA adapters",
    icon: <CpuIcon className="size-5 text-indigo-400" />,
    tag: "Encoders",
    details: {
      overview:
        "Extracts dense representation vectors using a Vision Transformer (ViT) with dynamic central-wavelength positional embeddings. Employs low-rank adaptation (LoRA) on attention weights, keeping 99.74% of parameters frozen.",
      techStack: ["DOFA ViT-Base", "LoRA Adapters (r=8)", "Wavelength Hypernet", "Multi-Label Jaccard"],
      metrics: [
        { label: "Embedding Dim", value: "768-D" },
        { label: "Trainable Params", value: "0.26% (294.9K)" },
        { label: "Encoder Latency", value: "14.2ms" },
      ],
      formula: "E_{λ} = ViT_{frozen}(x) + HyperNet(λ_{center})",
      flowInput: "Band Reflectance Tensors + Wavelengths",
      flowOutput: "Hypersphere Modality Embeddings (768-D)",
    },
  },
  {
    id: "cfm-bridge",
    number: "03",
    title: "CFM Latent ODE Bridge",
    shortDesc: "Continuous Flow Matching ODE translating SAR to Optical latent space",
    icon: <NetworkIcon className="size-5 text-amber-400" />,
    tag: "Latent Flow",
    details: {
      overview:
        "Solves a neural vector-field Ordinary Differential Equation (ODE) using Conditional Flow Matching (CFM). Computes straight-line probability transport paths between SAR radar embeddings and Optical multi-spectral embeddings without structural loss.",
      techStack: ["TorchDiffEq ODE", "Continuous Flow Matching", "Optimal Transport", "Cosine Metric Loss"],
      metrics: [
        { label: "ODE Integration", value: "8 Euler Steps" },
        { label: "Bridge Latency", value: "11.6ms" },
        { label: "Alignment mAP", value: "91.49%" },
      ],
      formula: "dx/dt = v_t(x; θ), \\quad x(0) = z_{SAR}, \\quad x(1) = z_{Optical}",
      flowInput: "SAR Latent Vector (z_SAR)",
      flowOutput: "Bridged Optical Latent Vector (z_Opt)",
    },
  },
  {
    id: "faiss-search",
    number: "04",
    title: "FAISS Index & Reranking",
    shortDesc: "Sub-30ms similarity search & k-reciprocal land-cover graph re-ranking",
    icon: <DatabaseIcon className="size-5 text-emerald-400" />,
    tag: "Vector Search",
    details: {
      overview:
        "Indexes normalized hypersphere vectors into an inverted-file product quantization (IVF-PQ) index. Executes sub-30ms Approximate Nearest Neighbor (ANN) search with k-reciprocal contextual re-ranking based on multi-label land cover overlap.",
      techStack: ["FAISS IndexIVFPQ", "Cosine Hypersphere", "k-Reciprocal Re-ranking", "Multi-Label Jaccard"],
      metrics: [
        { label: "Search Latency", value: "< 3.2ms" },
        { label: "Total E2E Pipeline", value: "28.9ms" },
        { label: "Recall@10", value: "92.4%" },
      ],
      flowInput: "Bridged Latent Query Vector",
      flowOutput: "Ranked Candidate Scenes + Geo Metadata",
    },
  },
];

export default function Architecture() {
  const [activeStepId, setActiveStepId] = useState<string>("cfm-bridge");
  const activeStep =
    ARCHITECTURE_STEPS.find((s) => s.id === activeStepId) ||
    ARCHITECTURE_STEPS[2];

  return (
    <Section id="architecture" className="relative overflow-hidden py-16 sm:py-24">
      {/* Subtle Background Glow */}
      <Glow variant="center" className="opacity-40" />

      <div className="max-w-container mx-auto flex flex-col gap-12 px-4">
        {/* Section Header */}
        <div className="flex flex-col items-center gap-4 text-center">
          <Badge variant="outline" className="border-brand/40 text-brand">
            SABER System Architecture & Cross-Modal Flow
          </Badge>
          <h2 className="text-3xl font-bold tracking-tight sm:text-5xl max-w-[800px]">
            End-to-End Pipeline Architecture
          </h2>
          <p className="text-muted-foreground text-sm sm:text-lg max-w-[720px]">
            Unifying Synthetic Aperture Radar (SAR) and Optical Multispectral satellite imagery onto a single metric-optimised hypersphere via CFM ODE latent transport.
          </p>
        </div>

        {/* Pipeline Stepper Buttons */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {ARCHITECTURE_STEPS.map((step) => {
            const isActive = step.id === activeStepId;
            return (
              <button
                key={step.id}
                onClick={() => setActiveStepId(step.id)}
                className={cn(
                  "group relative flex flex-col gap-3 rounded-xl border p-5 text-left transition-all duration-200 select-none cursor-pointer focus:outline-none focus-visible:outline-none focus:ring-0 focus-visible:ring-0",
                  isActive
                    ? "border-border/60 bg-brand/10"
                    : "border-border/60 bg-card/50 hover:bg-card/80",
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-muted-foreground group-hover:text-foreground">
                    STAGE {step.number}
                  </span>
                  <Badge variant={isActive ? "brand" : "outline"} size="sm">
                    {step.tag}
                  </Badge>
                </div>
                <div className="flex items-center gap-3">
                  <div
                    className={cn(
                      "flex size-9 items-center justify-center rounded-lg border transition-colors",
                      isActive
                        ? "border-brand/50 bg-brand/20"
                        : "border-border bg-background/50",
                    )}
                  >
                    {step.icon}
                  </div>
                  <h3 className="font-semibold text-sm sm:text-base leading-tight">
                    {step.title}
                  </h3>
                </div>
                <p className="text-xs text-muted-foreground line-clamp-2 leading-snug">
                  {step.shortDesc}
                </p>
                {isActive && (
                  <div className="bg-brand absolute -bottom-px left-4 right-4 h-0.5 rounded-full" />
                )}
              </button>
            );
          })}
        </div>

        {/* Interactive Architecture Flow View */}
        <div className="relative rounded-2xl border border-border/80 bg-card/60 p-6 backdrop-blur-xl sm:p-8">
          {/* Top Flow Ribbon */}
          <div className="mb-8 flex flex-wrap items-center justify-between gap-4 border-b border-border/50 pb-6">
            <div className="flex items-center gap-3">
              <span className="flex size-7 items-center justify-center rounded-full bg-brand/20 text-brand text-xs font-bold font-mono">
                {activeStep.number}
              </span>
              <div>
                <h3 className="text-xl font-bold">{activeStep.title}</h3>
                <span className="text-xs text-muted-foreground">
                  Module Pipeline Stage {activeStep.number} of 04
                </span>
              </div>
            </div>

            {/* Input -> Output Pipeline pill */}
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-background/60 p-2 text-xs font-mono">
              <span className="text-muted-foreground">{activeStep.details.flowInput}</span>
              <ArrowRightIcon className="size-3.5 text-brand" />
              <span className="font-semibold text-foreground">{activeStep.details.flowOutput}</span>
            </div>
          </div>

          {/* Details Content Grid */}
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
            {/* Description & Overview */}
            <div className="flex flex-col gap-6 lg:col-span-7">
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                  Stage Overview
                </h4>
                <p className="text-sm sm:text-base text-foreground/90 leading-relaxed">
                  {activeStep.details.overview}
                </p>
              </div>

              {activeStep.details.formula && (
                <div className="rounded-xl border border-brand/30 bg-brand/5 p-4 font-mono text-xs sm:text-sm text-brand-foreground">
                  <div className="text-[10px] text-muted-foreground uppercase font-sans mb-1">
                    Mathematical Formulation
                  </div>
                  <code>{activeStep.details.formula}</code>
                </div>
              )}

              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
                  Core Technologies & Components
                </h4>
                <div className="flex flex-wrap gap-2">
                  {activeStep.details.techStack.map((tech) => (
                    <span
                      key={tech}
                      className="inline-flex items-center gap-1.5 rounded-md border border-border/80 bg-background/80 px-3 py-1.5 text-xs font-medium text-foreground"
                    >
                      <GitCommitIcon className="size-3 text-brand" />
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Performance Telemetry Card */}
            <div className="flex flex-col justify-between gap-6 rounded-xl border border-border bg-background/70 p-6 lg:col-span-5">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Performance Telemetry
                  </h4>
                  <ZapIcon className="size-4 text-amber-400 animate-pulse" />
                </div>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:grid-cols-1">
                  {activeStep.details.metrics.map((metric) => (
                    <div
                      key={metric.label}
                      className="flex items-center justify-between rounded-lg border border-border/40 bg-card/40 p-3"
                    >
                      <span className="text-xs text-muted-foreground font-medium">
                        {metric.label}
                      </span>
                      <span className="font-mono text-sm font-bold text-brand">
                        {metric.value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <Button asChild size="default" className="w-full gap-2 mt-4">
                <a href="/dashboard/format/query">
                  Test Module in Live Dashboard
                  <ArrowRightIcon className="size-4" />
                </a>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}
