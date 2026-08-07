import { Satellite } from "lucide-react";
import { cn } from "@/lib/utils";
import { ModeToggle } from "../../ui/mode-toggle";
import { Footer, FooterBottom, FooterColumn, FooterContent } from "../../ui/footer";

export default function FooterSection({ className }: { className?: string }) {
  return (
    <footer className={cn("bg-background w-full px-4", className)}>
      <div className="max-w-container mx-auto">
        <Footer>
          <FooterContent>
            <FooterColumn className="col-span-2 sm:col-span-3 md:col-span-1">
              <div className="flex items-center gap-2.5">
                <div className="flex items-center justify-center size-8 rounded-lg bg-[#FBBA72]/15 border border-[#FBBA72]/30">
                  <Satellite className="size-4 text-[#FBBA72]" />
                </div>
                <div>
                  <div className="font-bold text-base">SABER</div>
                  <div className="text-xs text-muted-foreground">ISRO BAH 2026 · PS-11</div>
                </div>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed mt-3 max-w-[220px]">
                Cross-modal satellite image retrieval at sub-30ms. Built by Team Sentinel8.
              </p>
            </FooterColumn>
            <FooterColumn>
              <h3 className="text-sm font-semibold pt-1">Project</h3>
              <a href="/dashboard/format/query" className="text-muted-foreground text-sm hover:text-foreground transition-colors">Live Demo</a>
              <a href="#benchmarks" className="text-muted-foreground text-sm hover:text-foreground transition-colors">Benchmarks</a>
              <a href="#use-cases" className="text-muted-foreground text-sm hover:text-foreground transition-colors">Use Cases</a>
              <a href="#faq" className="text-muted-foreground text-sm hover:text-foreground transition-colors">FAQ</a>
            </FooterColumn>
            <FooterColumn>
              <h3 className="text-sm font-semibold pt-1">Tech Stack</h3>
              <span className="text-muted-foreground text-sm">PyTorch 2.13</span>
              <span className="text-muted-foreground text-sm">DOFA ViT-Base</span>
              <span className="text-muted-foreground text-sm">FAISS · FastAPI</span>
              <span className="text-muted-foreground text-sm">Next.js 16 · Tailwind</span>
            </FooterColumn>
            <FooterColumn>
              <h3 className="text-sm font-semibold pt-1">Links</h3>
              <a href="https://github.com/SK8-infi/SABER" target="_blank" rel="noopener noreferrer" className="text-muted-foreground text-sm hover:text-foreground transition-colors">GitHub</a>
              <a href="https://www.isro.gov.in" target="_blank" rel="noopener noreferrer" className="text-muted-foreground text-sm hover:text-foreground transition-colors">ISRO</a>
              <a href="https://sentinel.esa.int" target="_blank" rel="noopener noreferrer" className="text-muted-foreground text-sm hover:text-foreground transition-colors">ESA Sentinel</a>
            </FooterColumn>
          </FooterContent>
          <FooterBottom>
            <div className="text-sm text-muted-foreground">© 2026 Team Sentinel8 · SABER · ISRO BAH Grand Finale</div>
            <div className="flex items-center gap-4">
              <span className="text-xs text-muted-foreground">Copernicus Sentinel data © ESA 2023–2026</span>
              <ModeToggle />
            </div>
          </FooterBottom>
        </Footer>
      </div>
    </footer>
  );
}
