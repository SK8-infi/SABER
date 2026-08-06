import { Section } from "../../ui/section";
import { Badge } from "../../ui/badge";

const TECH_STACK = [
  { name: "PyTorch", version: "2.13", desc: "Deep Learning" },
  { name: "DOFA ViT", version: "Base", desc: "Backbone" },
  { name: "FastAPI", version: "0.115", desc: "Backend" },
  { name: "FAISS", version: "1.8", desc: "Vector Search" },
  { name: "Next.js", version: "16.2", desc: "Frontend" },
  { name: "Sentinel-1/2", version: "ESA", desc: "Datasets" },
];

export default function Logos() {
  return (
    <Section>
      <div className="max-w-container mx-auto flex flex-col items-center gap-8 text-center">
        <div className="flex flex-col items-center gap-3">
          <Badge variant="outline" className="border-[#FBBA72]/30 text-[#FBBA72]">
            Team Sentinel8 · ISRO BAH 2026 Grand Finale
          </Badge>
          <h2 className="text-md font-semibold sm:text-2xl text-muted-foreground">
            Built on proven open-science infrastructure
          </h2>
        </div>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {TECH_STACK.map(t => (
            <div key={t.name}
              className="flex items-center gap-2.5 rounded-full border border-border/60 bg-muted/10 px-4 py-2 hover:border-[#FBBA72]/40 transition-colors">
              <div className="size-2 rounded-full bg-[#FBBA72]" />
              <span className="text-sm font-semibold text-foreground">{t.name}</span>
              <span className="text-xs text-muted-foreground font-mono">{t.version}</span>
              <span className="text-xs text-muted-foreground border-l border-border/60 pl-2">{t.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}
