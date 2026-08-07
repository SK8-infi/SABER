"use client";

import { useState } from "react";
import { Section } from "../../ui/section";
import { Badge } from "../../ui/badge";
import { Card, CardContent } from "../../ui/card";
import { cn } from "@/lib/utils";
import {
  CloudOff,
  Layers,
  Search,
  ShieldCheck,
  Sprout,
  MapPin,
  CheckCircle2,
  Sparkles,
} from "lucide-react";

const USE_CASES = [
  {
    id: "flood",
    icon: <CloudOff className="size-4" />,
    tag: "Disaster Response",
    title: "All-Weather Flood Mapping",
    description:
      "On 10 Sep 2023, Storm Daniel caused catastrophic flooding in Derna, Libya. Dense cloud cover blocked every optical satellite for days. Sentinel-1 SAR cut right through — imaging the flood extent in full clarity. SABER takes that SAR image as the query and retrieves the closest pre-flood Sentinel-2 optical reference from the archive in 28ms, giving responders an instant before/after pair for damage assessment — no clear sky required.",
    sarLabel: "Sentinel-1 SAR · Flood Inundation Query (ESA/Copernicus)",
    opticalLabel: "Sentinel-2 Optical · Pre-flood Reference Match (ESA)",
    sarImg: "/images/satellite/demo_4_0.png",
    opticalImg: "/images/satellite/demo_18_0.png",
    result: "Matched pre-disaster optical scene in 28.48ms",
  },
  {
    id: "crop",
    icon: <Sprout className="size-4" />,
    tag: "Agriculture",
    title: "Crop Monitoring Through Monsoon",
    description:
      "During India's kharif season, cloud cover makes optical imagery unavailable for months. SABER queries available SAR data and retrieves the semantically closest optical scene — enabling crop-type classification and yield estimation without waiting for clear skies.",
    sarLabel: "Sentinel-1 SAR · Central Farmland Radar (ESA)",
    opticalLabel: "Sentinel-2 Optical · Crop Canopy Reference (ESA)",
    sarImg: "/images/satellite/demo_4_0.png",
    opticalImg: "/images/satellite/demo_11_1.png",
    result: "Cross-modal F1@5 = 73.51% on BEN-14K benchmark",
  },
  {
    id: "urban",
    icon: <MapPin className="size-4" />,
    tag: "Urban Change Detection",
    title: "Construction & Urban Expansion",
    description:
      "SAR detects new construction and urban expansion even at night or through haze. SABER retrieves the closest optical reference from the historical archive, letting analysts instantly compare land-cover changes across time and sensors.",
    sarLabel: "Sentinel-1 SAR · Urban Fabric Radar (ESA)",
    opticalLabel: "Sentinel-2 Optical · Historical Urban Match (ESA)",
    sarImg: "/images/satellite/demo_4_0.png",
    opticalImg: "/images/satellite/demo_18_0.png",
    result: "Sub-28.5ms retrieval from 14,832-scene gallery",
  },
  {
    id: "defense",
    icon: <ShieldCheck className="size-4" />,
    tag: "Defense & Surveillance",
    title: "All-Weather Target Identification",
    description:
      "SAR imagery is interpretable by machines but hard for analysts unfamiliar with radar backscatter. SABER bridges SAR embeddings to visually interpretable optical matches — enabling rapid confirmation of targets from the historical archive.",
    sarLabel: "Sentinel-1 SAR · Target Surveillance Scene (ESA)",
    opticalLabel: "Sentinel-2 Optical · Visual Optical Match (ESA)",
    sarImg: "/images/satellite/demo_4_0.png",
    opticalImg: "/images/satellite/demo_11_1.png",
    result: "Cross-Modal mAP = 91.49% (24% higher vs RemoteCLIP)",
  },
  {
    id: "archive",
    icon: <Search className="size-4" />,
    tag: "Archive Search",
    title: "Cross-Sensor Archive Retrieval",
    description:
      "Space agencies hold petabyte-scale archives spanning decades and multiple sensors. SABER lets researchers search millions of scenes in real-time using any sensor as query — semantic image search across the entire Earth observation archive.",
    sarLabel: "Sentinel-1 SAR · Archive Radar Query (ESA)",
    opticalLabel: "Sentinel-2 Optical · Matched Archive Scene (ESA)",
    sarImg: "/images/satellite/demo_4_0.png",
    opticalImg: "/images/satellite/demo_18_0.png",
    result: "14,832-scene gallery · 768-dim embeddings",
  },
  {
    id: "forest",
    icon: <Layers className="size-4" />,
    tag: "Environmental Monitoring",
    title: "Deforestation & Fire Scar Detection",
    description:
      "Forest fire scars and deforestation appear as strong SAR texture anomalies. SABER retrieves the matching optical reference scene for each detected change, letting forest departments confirm, document and measure affected area with visual clarity.",
    sarLabel: "Sentinel-1 SAR · Deforestation Scar (ESA)",
    opticalLabel: "Sentinel-2 Optical · Forest Canopy Match (ESA)",
    sarImg: "/images/satellite/demo_4_0.png",
    opticalImg: "/images/satellite/demo_11_1.png",
    result: "0.26% Trainable Params (294.9K / 111.6M)",
  },
];

function ImagePair({
  sarImg,
  opticalImg,
  sarLabel,
  opticalLabel,
}: {
  sarImg: string;
  opticalImg: string;
  sarLabel: string;
  opticalLabel: string;
}) {
  return (
    <div className="grid grid-cols-2 gap-3 rounded-2xl overflow-hidden p-1.5 bg-card/60 border border-border/60">
      {/* SAR Image Container */}
      <div className="relative flex flex-col gap-2">
        <div className="relative aspect-square rounded-xl overflow-hidden bg-zinc-950 border border-border/60 group">
          <img
            src={sarImg}
            alt={sarLabel}
            className="w-full h-full object-cover grayscale contrast-125 brightness-90 group-hover:scale-105 transition-transform duration-300"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
          <Badge
            variant="outline"
            className="absolute bottom-2 left-2 text-[9px] font-bold font-mono uppercase tracking-wider text-sky-400 bg-background/80 border-sky-500/40 backdrop-blur-md px-2 py-0.5"
          >
            SAR · Sees Through Clouds
          </Badge>
        </div>
        <p className="text-[10px] text-muted-foreground font-sans leading-snug px-1 truncate" title={sarLabel}>
          {sarLabel}
        </p>
      </div>

      {/* Optical Image Container */}
      <div className="relative flex flex-col gap-2">
        <div className="relative aspect-square rounded-xl overflow-hidden bg-zinc-950 border border-[#FBBA72]/40 group">
          <img
            src={opticalImg}
            alt={opticalLabel}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
          <Badge
            variant="outline"
            className="absolute bottom-2 left-2 text-[9px] font-bold font-mono uppercase tracking-wider text-[#FBBA72] bg-background/80 border-[#FBBA72]/50 backdrop-blur-md px-2 py-0.5"
          >
            OPTICAL · Retrieved Match
          </Badge>
        </div>
        <p className="text-[10px] text-muted-foreground font-sans leading-snug px-1 truncate" title={opticalLabel}>
          {opticalLabel}
        </p>
      </div>
    </div>
  );
}

export default function UseCases() {
  const [activeId, setActiveId] = useState("flood");
  const active = USE_CASES.find((u) => u.id === activeId) ?? USE_CASES[0];

  return (
    <Section>
      <div className="max-w-container mx-auto flex flex-col gap-10 sm:gap-14 font-sans">
        {/* Header Section */}
        <div className="flex flex-col items-center gap-3 text-center">
          <Badge
            variant="outline"
            className="border-[#FBBA72]/40 text-[#FBBA72] bg-[#FBBA72]/10 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider"
          >
            <Sparkles className="size-3 mr-1 text-[#FBBA72]" />
            Real-World Applications
          </Badge>
          <h2 className="text-3xl font-bold tracking-tight sm:text-5xl leading-tight max-w-[640px] text-foreground font-sans">
            Where SABER changes the game
          </h2>
          <p className="text-muted-foreground max-w-[600px] text-sm sm:text-base leading-relaxed font-sans">
            SAR radar sees through clouds, smoke, and darkness. Optical imagery is visually interpretable. SABER bridges both — enabling cross-sensor retrieval in under 30ms.
          </p>
        </div>

        {/* Tab Selector Chips */}
        <div className="flex flex-wrap justify-center gap-2">
          {USE_CASES.map((uc) => {
            const isActive = activeId === uc.id;
            return (
              <button
                key={uc.id}
                onClick={() => setActiveId(uc.id)}
                className={cn(
                  "flex items-center gap-2 rounded-full border px-4 py-2 text-xs sm:text-sm font-semibold transition-all duration-200 cursor-pointer font-sans",
                  isActive
                    ? "border-[#FBBA72]/60 bg-[#FBBA72]/15 text-[#FBBA72] shadow-sm"
                    : "border-border/60 bg-card/40 text-muted-foreground hover:border-border/80 hover:text-foreground hover:bg-card/70",
                )}
              >
                {uc.icon}
                <span>{uc.tag}</span>
              </button>
            );
          })}
        </div>

        {/* Main Content Card */}
        <Card className="border-border/60 bg-card/60 backdrop-blur-xl shadow-lg rounded-3xl overflow-hidden p-6 sm:p-8">
          <CardContent className="p-0">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
              {/* Left Column: Text Info */}
              <div className="flex flex-col gap-5">
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className="border-[#FBBA72]/40 text-[#FBBA72] bg-[#FBBA72]/10 text-xs font-semibold px-3 py-0.5 rounded-full"
                  >
                    {active.tag}
                  </Badge>
                </div>

                <h3 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground font-sans leading-snug">
                  {active.title}
                </h3>

                <p className="text-muted-foreground text-sm sm:text-base leading-relaxed font-sans">
                  {active.description}
                </p>

                {/* How SABER Solves This Box */}
                <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-muted/20 p-5 font-sans">
                  <div className="flex items-center gap-2">
                    <div className="size-2 rounded-full bg-[#FBBA72]" />
                    <span className="text-xs font-bold uppercase tracking-wider text-[#FBBA72]">
                      How SABER Solves This
                    </span>
                  </div>
                  <ol className="flex flex-col gap-2.5 text-xs sm:text-sm text-muted-foreground">
                    <li className="flex items-start gap-2.5">
                      <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-[#FBBA72]/20 border border-[#FBBA72]/40 text-[11px] font-bold text-[#FBBA72] font-mono">
                        1
                      </span>
                      <span>SAR image encoded by wavelength-conditioned DOFA ViT backbone</span>
                    </li>
                    <li className="flex items-start gap-2.5">
                      <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-[#FBBA72]/20 border border-[#FBBA72]/40 text-[11px] font-bold text-[#FBBA72] font-mono">
                        2
                      </span>
                      <span>CFM latent bridge transports SAR embedding → optical hypersphere</span>
                    </li>
                    <li className="flex items-start gap-2.5">
                      <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-[#FBBA72]/20 border border-[#FBBA72]/40 text-[11px] font-bold text-[#FBBA72] font-mono">
                        3
                      </span>
                      <span>FAISS cosine search retrieves Top-K optical matches in &lt;1ms</span>
                    </li>
                  </ol>
                </div>

                {/* Result Pill */}
                <div className="flex items-center gap-2 pt-1">
                  <CheckCircle2 className="size-4 text-emerald-400 shrink-0" />
                  <span className="text-sm font-semibold font-mono text-[#FBBA72]">
                    {active.result}
                  </span>
                </div>
              </div>

              {/* Right Column: Image Pair & Transfer Path */}
              <div className="flex flex-col gap-4">
                <ImagePair
                  sarImg={active.sarImg}
                  opticalImg={active.opticalImg}
                  sarLabel={active.sarLabel}
                  opticalLabel={active.opticalLabel}
                />
                <div className="flex items-center justify-center gap-3 text-xs text-muted-foreground p-2 rounded-xl bg-muted/20 border border-border/40 font-mono">
                  <span className="text-foreground/80 font-semibold">SAR Query</span>
                  <div className="flex items-center gap-1.5">
                    <div className="h-px w-6 bg-[#FBBA72]/60" />
                    <Badge
                      variant="outline"
                      className="border-[#FBBA72]/50 text-[#FBBA72] bg-[#FBBA72]/10 text-[10px] font-bold px-2 py-0.5"
                    >
                      CFM Bridge
                    </Badge>
                    <div className="h-px w-6 bg-[#FBBA72]/60" />
                  </div>
                  <span className="text-[#FBBA72] font-semibold">Retrieved Optical</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Bottom Metrics Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
          {[
            { val: "<28.5ms", label: "retrieval latency" },
            { val: "73.51%", label: "cross-modal F1@5" },
            { val: "91.49%", label: "cross-modal mAP" },
            { val: "0.26%", label: "trainable params (LoRA)" },
          ].map((s) => (
            <Card key={s.label} className="border-border/60 bg-card/40 backdrop-blur-md shadow-sm p-4 text-center rounded-2xl">
              <CardContent className="p-0 flex flex-col gap-1">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#FBBA72] tracking-tight font-sans">
                  {s.val}
                </span>
                <span className="text-xs uppercase tracking-wider text-muted-foreground font-semibold font-sans">
                  {s.label}
                </span>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </Section>
  );
}
