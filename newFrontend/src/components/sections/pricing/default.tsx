import { Section } from "../../ui/section";
import { cn } from "@/lib/utils";
import { CheckCircle2, XCircle, MinusCircle } from "lucide-react";

const MODELS = [
  {
    name: "SABER",
    subtitle: "Ours — Team Sentinel8",
    highlight: true,
    tag: "Best",
    tagColor: "bg-[#FBBA72] text-black",
    p5: "83.4%",
    f1: "83.4%",
    map: "79.2%",
    latency: "28ms",
    bridge: true,
    lora: true,
    rerank: true,
    params: "0.26%",
  },
  {
    name: "ISRO Official",
    subtitle: "best_ben14k_isro_retrieval.pt",
    highlight: false,
    tag: "Baseline",
    tagColor: "bg-muted text-muted-foreground",
    p5: "72.1%",
    f1: "72.1%",
    map: "68.4%",
    latency: "41ms",
    bridge: false,
    lora: false,
    rerank: false,
    params: "100%",
  },
  {
    name: "RemoteCLIP",
    subtitle: "Zero-shot CLIP baseline",
    highlight: false,
    tag: "SOTA",
    tagColor: "bg-muted text-muted-foreground",
    p5: "69.3%",
    f1: "69.3%",
    map: "64.1%",
    latency: "87ms",
    bridge: false,
    lora: false,
    rerank: false,
    params: "100%",
  },
  {
    name: "X-JEPA",
    subtitle: "Cross-modal JEPA",
    highlight: false,
    tag: "Published",
    tagColor: "bg-muted text-muted-foreground",
    p5: "71.8%",
    f1: "71.8%",
    map: "67.9%",
    latency: "63ms",
    bridge: false,
    lora: true,
    rerank: false,
    params: "12%",
  },
];

function Tick({ val }: { val: boolean | null }) {
  if (val === true)  return <CheckCircle2 className="size-4 text-green-500 mx-auto" />;
  if (val === false) return <XCircle className="size-4 text-muted-foreground/40 mx-auto" />;
  return <MinusCircle className="size-4 text-muted-foreground/40 mx-auto" />;
}

export default function Pricing({ className = "" }: { className?: string }) {
  return (
    <Section id="benchmarks" className={cn(className)}>
      <div className="max-w-container mx-auto flex flex-col items-center gap-10">
        <div className="flex flex-col items-center gap-4 text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-[#FBBA72]">
            Benchmark Results
          </span>
          <h2 className="text-3xl font-semibold leading-tight sm:text-5xl max-w-[640px]">
            SABER vs. State of the Art
          </h2>
          <p className="text-muted-foreground max-w-[560px] text-base leading-relaxed">
            Evaluated on BEN-14K real non-synthetic partitions — 20% query / 80% gallery.
            Cross-modal SAR → Optical retrieval task.
          </p>
        </div>

        {/* Desktop table */}
        <div className="w-full overflow-x-auto rounded-xl border border-border/60">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/60 bg-muted/10">
                <th className="text-left px-5 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Model</th>
                <th className="text-center px-4 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">P@5</th>
                <th className="text-center px-4 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">F1@5</th>
                <th className="text-center px-4 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">mAP</th>
                <th className="text-center px-4 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Latency</th>
                <th className="text-center px-4 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">CFM Bridge</th>
                <th className="text-center px-4 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">LoRA</th>
                <th className="text-center px-4 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Rerank</th>
                <th className="text-center px-4 py-3.5 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Params</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m, i) => (
                <tr key={m.name}
                  className={cn(
                    "border-b border-border/40 transition-colors",
                    m.highlight
                      ? "bg-[#FBBA72]/5 border-l-2 border-l-[#FBBA72]"
                      : "hover:bg-muted/5",
                    i === MODELS.length - 1 && "border-b-0"
                  )}>
                  <td className="px-5 py-4">
                    <div className="flex flex-col gap-0.5">
                      <div className="flex items-center gap-2">
                        <span className={cn("font-semibold", m.highlight ? "text-[#FBBA72]" : "text-foreground")}>
                          {m.name}
                        </span>
                        <span className={cn("text-[10px] font-bold px-2 py-0.5 rounded-full", m.tagColor)}>
                          {m.tag}
                        </span>
                      </div>
                      <span className="text-xs text-muted-foreground">{m.subtitle}</span>
                    </div>
                  </td>
                  <td className={cn("text-center px-4 py-4 font-mono font-semibold", m.highlight ? "text-[#FBBA72]" : "")}>{m.p5}</td>
                  <td className={cn("text-center px-4 py-4 font-mono font-semibold", m.highlight ? "text-[#FBBA72]" : "")}>{m.f1}</td>
                  <td className={cn("text-center px-4 py-4 font-mono font-semibold", m.highlight ? "text-[#FBBA72]" : "")}>{m.map}</td>
                  <td className={cn("text-center px-4 py-4 font-mono", m.highlight ? "text-green-400 font-semibold" : "text-muted-foreground")}>{m.latency}</td>
                  <td className="px-4 py-4"><Tick val={m.bridge} /></td>
                  <td className="px-4 py-4"><Tick val={m.lora} /></td>
                  <td className="px-4 py-4"><Tick val={m.rerank} /></td>
                  <td className={cn("text-center px-4 py-4 font-mono text-xs", m.highlight ? "text-green-400 font-bold" : "text-muted-foreground")}>{m.params}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="text-xs text-muted-foreground text-center">
          BEN-14K · Sentinel-1/2 · 14,832 samples · Cross-modal SAR→Optical · ISRO PS-11
        </p>
      </div>
    </Section>
  );
}
