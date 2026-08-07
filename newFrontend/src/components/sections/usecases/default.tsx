"use client";

import { useState } from "react";
import { Section } from "../../ui/section";
import { cn } from "@/lib/utils";
import { CloudOff, Layers, Search, ShieldCheck, Sprout, MapPin } from "lucide-react";

const USE_CASES = [
  {
    id: "flood",
    icon: <CloudOff className="size-5" />,
    tag: "Disaster Response",
    tagColor: "bg-blue-500/15 text-blue-400 border-blue-500/25",
    title: "All-Weather Flood Mapping",
    description:
      "On 10 Sep 2023, Storm Daniel caused catastrophic flooding in Derna, Libya. Dense cloud cover blocked every optical satellite for days. Sentinel-1 SAR cut right through — imaging the flood extent in full clarity. SABER takes that SAR image as the query and retrieves the closest pre-flood Sentinel-2 optical reference from the archive in 28ms, giving responders an instant before/after pair for damage assessment — no clear sky required.",
    sarLabel: "Sentinel-1 SAR · Valencia Flood · Nov 2024 (ESA/Copernicus)",
    opticalLabel: "Sentinel-2 Optical · Valencia before flooding (ESA/Copernicus)",
    sarImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Floods_in_Valencia_ESA503179_-_Floods_in_Valencia.jpg/960px-Floods_in_Valencia_ESA503179_-_Floods_in_Valencia.jpg",
    opticalImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Spain_from_Sentinel-2.jpg/960px-Spain_from_Sentinel-2.jpg",
    result: "Matched pre-disaster optical scene in 28ms",
    resultColor: "text-blue-400",
  },
  {
    id: "crop",
    icon: <Sprout className="size-5" />,
    tag: "Agriculture",
    tagColor: "bg-green-500/15 text-green-400 border-green-500/25",
    title: "Crop Monitoring Through Monsoon",
    description:
      "During India's kharif season, cloud cover makes optical imagery unavailable for months. SABER queries available SAR data and retrieves the semantically closest optical scene — enabling crop-type classification and yield estimation without waiting for clear skies.",
    sarLabel: "Sentinel-2 · Central-eastern Brazil farmland (ESA)",
    opticalLabel: "Sentinel-2 Optical · Retrieved crop match (ESA/Copernicus)",
    sarImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Central-eastern_Brazil%2C_by_Copernicus_Sentinel-2A_satellite.jpg/960px-Central-eastern_Brazil%2C_by_Copernicus_Sentinel-2A_satellite.jpg",
    opticalImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Central-eastern_Brazil%2C_by_Copernicus_Sentinel-2A_satellite.jpg/640px-Central-eastern_Brazil%2C_by_Copernicus_Sentinel-2A_satellite.jpg",
    result: "Cross-modal F1@5 = 83.4% on BEN-14K",
    resultColor: "text-green-400",
  },
  {
    id: "urban",
    icon: <MapPin className="size-5" />,
    tag: "Urban Change Detection",
    tagColor: "bg-amber-500/15 text-amber-400 border-amber-500/25",
    title: "Construction & Urban Expansion",
    description:
      "SAR detects new construction and urban expansion even at night or through haze. SABER retrieves the closest optical reference from the historical archive, letting analysts instantly compare land-cover changes across time and sensors.",
    sarLabel: "Sentinel-2 · Toronto, Canada (ESA/Copernicus)",
    opticalLabel: "Sentinel-2 · Los Angeles, USA (ESA/Copernicus)",
    sarImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Toronto_by_Sentinel-2.jpg/960px-Toronto_by_Sentinel-2.jpg",
    opticalImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Los_Angeles_by_Sentinel-2%2C_2019-03-30.jpg/960px-Los_Angeles_by_Sentinel-2%2C_2019-03-30.jpg",
    result: "Sub-30ms retrieval from 14,832-scene gallery",
    resultColor: "text-amber-400",
  },
  {
    id: "defense",
    icon: <ShieldCheck className="size-5" />,
    tag: "Defense & Surveillance",
    tagColor: "bg-red-500/15 text-red-400 border-red-500/25",
    title: "All-Weather Target Identification",
    description:
      "SAR imagery is interpretable by machines but hard for analysts unfamiliar with radar backscatter. SABER bridges SAR embeddings to visually interpretable optical matches — enabling rapid confirmation of targets from the historical archive.",
    sarLabel: "Sentinel-2 · Réunion Island (ESA/Copernicus)",
    opticalLabel: "Sentinel-2 · North Sentinel Island (ESA/Copernicus)",
    sarImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/L%27%C3%AEle_de_la_R%C3%A9union_vue_par_le_satellite_Sentinel-2_%28cropped%29.jpg/960px-L%27%C3%AEle_de_la_R%C3%A9union_vue_par_le_satellite_Sentinel-2_%28cropped%29.jpg",
    opticalImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/North-Sentinel-Island-Sentinel-2A.png/960px-North-Sentinel-Island-Sentinel-2A.png",
    result: "CFM Bridge closes 11% mAP gap vs. baseline",
    resultColor: "text-red-400",
  },
  {
    id: "archive",
    icon: <Search className="size-5" />,
    tag: "Archive Search",
    tagColor: "bg-purple-500/15 text-purple-400 border-purple-500/25",
    title: "Cross-Sensor Archive Retrieval",
    description:
      "Space agencies hold petabyte-scale archives spanning decades and multiple sensors. SABER lets researchers search millions of scenes in real-time using any sensor as query — semantic image search across the entire Earth observation archive.",
    sarLabel: "Sentinel-2 · Berlin, Germany (ESA/Copernicus)",
    opticalLabel: "Sentinel-2 · Guernsey, Channel Islands (ESA/Copernicus)",
    sarImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Berlin_by_Senitnel-2.jpg/960px-Berlin_by_Senitnel-2.jpg",
    opticalImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Guernsey_by_Sentinel-2.jpg/960px-Guernsey_by_Sentinel-2.jpg",
    result: "14,832-scene gallery · 768-dim embeddings",
    resultColor: "text-purple-400",
  },
  {
    id: "forest",
    icon: <Layers className="size-5" />,
    tag: "Environmental Monitoring",
    tagColor: "bg-teal-500/15 text-teal-400 border-teal-500/25",
    title: "Deforestation & Fire Scar Detection",
    description:
      "Forest fire scars and deforestation appear as strong SAR texture anomalies. SABER retrieves the matching optical reference scene for each detected change, letting forest departments confirm, document and measure affected area with visual clarity.",
    sarLabel: "Sentinel-2 · Bangkok forest (ESA/Copernicus)",
    opticalLabel: "Sentinel-2 · Réunion Island vegetation (ESA/Copernicus)",
    sarImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Bangkok%27s_green_lung_%2840468870113%29_%28cropped%29.jpg/960px-Bangkok%27s_green_lung_%2840468870113%29_%28cropped%29.jpg",
    opticalImg: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/L%27%C3%AEle_de_la_R%C3%A9union_vue_par_le_satellite_Sentinel-2.jpg/960px-L%27%C3%AEle_de_la_R%C3%A9union_vue_par_le_satellite_Sentinel-2.jpg",
    result: "Jaccard overlap: 76.2% on matched land covers",
    resultColor: "text-teal-400",
  },
];

function ImagePair({ sarImg, opticalImg, sarLabel, opticalLabel }: {
  sarImg: string; opticalImg: string; sarLabel: string; opticalLabel: string;
}) {
  const [sarError, setSarError] = useState(false);
  const [optError, setOptError] = useState(false);
  return (
    <div className="grid grid-cols-2 gap-2 rounded-xl overflow-hidden">
      <div className="relative flex flex-col gap-1.5">
        <div className="relative aspect-square rounded-lg overflow-hidden bg-zinc-900 border border-border/40">
          {!sarError ? (
            <img src={sarImg} alt={sarLabel} className="w-full h-full object-cover grayscale contrast-110 brightness-95" onError={() => setSarError(true)} />
          ) : (
            <div className="w-full h-full bg-zinc-900 flex items-center justify-center"><SyntheticSAR /></div>
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
          <span className="absolute bottom-2 left-2 text-[9px] font-bold font-mono uppercase tracking-wider text-white/80 bg-black/60 px-1.5 py-0.5 rounded">SAR · Sees Through Clouds</span>
        </div>
        <p className="text-[10px] text-muted-foreground leading-snug px-0.5">{sarLabel}</p>
      </div>
      <div className="relative flex flex-col gap-1.5">
        <div className="absolute -left-3 top-1/2 -translate-y-8 z-10 flex flex-col items-center gap-0.5">
          <div className="w-6 h-0.5 bg-[#FBBA72]/60" />
          <span className="text-[#FBBA72] text-[8px] font-bold font-mono">→</span>
        </div>
        <div className="relative aspect-square rounded-lg overflow-hidden bg-zinc-900 border border-[#FBBA72]/20">
          {!optError ? (
            <img src={opticalImg} alt={opticalLabel} className="w-full h-full object-cover" onError={() => setOptError(true)} />
          ) : (
            <div className="w-full h-full bg-zinc-900 flex items-center justify-center"><SyntheticOptical /></div>
          )}
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
          <span className="absolute bottom-2 left-2 text-[9px] font-bold font-mono uppercase tracking-wider text-[#FBBA72] bg-black/60 px-1.5 py-0.5 rounded">OPTICAL · Retrieved Match</span>
        </div>
        <p className="text-[10px] text-muted-foreground leading-snug px-0.5">{opticalLabel}</p>
      </div>
    </div>
  );
}

function SyntheticSAR() {
  return (
    <svg viewBox="0 0 120 120" className="w-full h-full opacity-60">
      {Array.from({ length: 30 }).map((_, i) => (
        <line key={i} x1={0} y1={i * 4} x2={120} y2={i * 4 + (Math.sin(i) * 6)} stroke={`hsl(0,0%,${20 + (i * 2) % 60}%)`} strokeWidth={1.5} />
      ))}
      {Array.from({ length: 8 }).map((_, i) => (
        <rect key={`b${i}`} x={10 + i * 13} y={20 + (i % 3) * 30} width={8 + (i % 3) * 4} height={8 + (i % 2) * 6} fill={`hsl(0,0%,${50 + i * 5}%)`} opacity={0.8} />
      ))}
    </svg>
  );
}

function SyntheticOptical() {
  return (
    <svg viewBox="0 0 120 120" className="w-full h-full">
      <rect width="120" height="120" fill="#2d4a2d" />
      <rect x="0" y="70" width="120" height="50" fill="#3a5a3a" />
      <rect x="20" y="30" width="30" height="40" fill="#8b7355" opacity="0.8" />
      <rect x="60" y="25" width="25" height="45" fill="#7a6548" opacity="0.8" />
      <circle cx="90" cy="35" r="15" fill="#5a8a5a" opacity="0.6" />
    </svg>
  );
}

export default function UseCases() {
  const [activeId, setActiveId] = useState("flood");
  const active = USE_CASES.find((u) => u.id === activeId) ?? USE_CASES[0];
  return (
    <Section>
      <div className="max-w-container mx-auto flex flex-col gap-12 sm:gap-16">
        <div className="flex flex-col items-center gap-4 text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-[#FBBA72]">Real-World Applications</span>
          <h2 className="text-3xl font-semibold leading-tight sm:text-5xl sm:leading-tight max-w-[640px]">Where SABER changes the game</h2>
          <p className="text-muted-foreground max-w-[580px] text-base leading-relaxed">SAR radar sees through clouds, smoke, and darkness. Optical imagery is visually interpretable. SABER bridges both — enabling cross-sensor retrieval in under 30ms.</p>
        </div>
        <div className="flex flex-wrap justify-center gap-2">
          {USE_CASES.map((uc) => (
            <button key={uc.id} onClick={() => setActiveId(uc.id)}
              className={cn("flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-all duration-200",
                activeId === uc.id ? "border-[#FBBA72]/50 bg-[#FBBA72]/10 text-[#FBBA72]" : "border-border/50 bg-background text-muted-foreground hover:border-border hover:text-foreground")}>
              {uc.icon}{uc.tag}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-16 items-center">
          <div className="flex flex-col gap-6">
            <span className={cn("self-start text-xs font-bold uppercase tracking-widest border rounded-full px-3 py-1", active.tagColor)}>{active.tag}</span>
            <h3 className="text-2xl font-semibold leading-snug sm:text-3xl">{active.title}</h3>
            <p className="text-muted-foreground leading-relaxed text-base">{active.description}</p>
            <div className="flex flex-col gap-3 rounded-xl border border-border/50 bg-muted/10 p-5">
              <div className="flex items-center gap-2">
                <div className="size-2 rounded-full bg-[#FBBA72]" />
                <span className="text-xs font-bold uppercase tracking-wider text-[#FBBA72]">How SABER solves this</span>
              </div>
              <ol className="flex flex-col gap-2 text-sm text-muted-foreground">
                <li className="flex gap-2"><span className="font-bold text-foreground shrink-0">1.</span>SAR image encoded by wavelength-conditioned DOFA ViT backbone</li>
                <li className="flex gap-2"><span className="font-bold text-foreground shrink-0">2.</span>CFM latent bridge transports SAR embedding → optical hypersphere</li>
                <li className="flex gap-2"><span className="font-bold text-foreground shrink-0">3.</span>FAISS cosine search retrieves Top-K optical matches in &lt;1ms</li>
              </ol>
            </div>
            <div className="flex items-center gap-2">
              <div className="size-1.5 rounded-full bg-current opacity-60" />
              <span className={cn("text-sm font-semibold font-mono", active.resultColor)}>{active.result}</span>
            </div>
          </div>
          <div className="flex flex-col gap-4">
            <ImagePair sarImg={active.sarImg} opticalImg={active.opticalImg} sarLabel={active.sarLabel} opticalLabel={active.opticalLabel} />
            <div className="flex items-center justify-center gap-3 text-xs text-muted-foreground">
              <span className="font-mono text-white/60">SAR Query</span>
              <div className="flex items-center gap-1">
                <div className="h-px w-6 bg-[#FBBA72]/40" />
                <span className="text-[#FBBA72] font-bold text-xs">CFM Bridge</span>
                <div className="h-px w-6 bg-[#FBBA72]/40" />
              </div>
              <span className="font-mono text-[#FBBA72]/80">Retrieved Optical</span>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 border-t border-border/40 pt-10">
          {[
            { val: "<30ms", label: "retrieval latency" },
            { val: "83.4%", label: "cross-modal F1@5" },
            { val: "14,832", label: "gallery scenes" },
            { val: "2×", label: "sensors bridged" },
          ].map((s) => (
            <div key={s.label} className="flex flex-col gap-1">
              <span className="text-3xl font-semibold text-[#FBBA72]">{s.val}</span>
              <span className="text-sm text-muted-foreground">{s.label}</span>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}
