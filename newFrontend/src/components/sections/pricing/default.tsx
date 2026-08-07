import { Terminal, Cpu, Database, GitFork } from "lucide-react";

import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";

import { Section } from "../../ui/section";

interface PricingProps {
  title?: string | false;
  description?: string | false;
  className?: string;
}

export default function Pricing({
  title = "Open-Source & Free Forever",
  description = "SABER is fully open-source under the MIT License. Clone, train, and deploy your own instance — no fees, no restrictions.",
  className = "",
}: PricingProps) {
  return (
    <Section className={cn(className)}>
      <div className="mx-auto flex max-w-5xl flex-col items-center gap-12">
        {(title || description) && (
          <div className="flex flex-col items-center gap-4 px-4 text-center sm:gap-8">
            {title && (
              <h2 className="text-3xl leading-tight font-semibold sm:text-5xl sm:leading-tight">
                {title}
              </h2>
            )}
            {description && (
              <p className="text-md text-muted-foreground max-w-[600px] font-medium sm:text-xl">
                {description}
              </p>
            )}
          </div>
        )}

        {/* Quick Start Cards */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 w-full px-4">
          {/* Clone Card */}
          <div className="rounded-2xl border border-border/60 bg-card/60 p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-[#FBBA72]/10 border border-[#FBBA72]/30 p-2.5">
                <GitFork className="size-5 text-[#FBBA72]" />
              </div>
              <h3 className="font-semibold text-base">1. Clone Repository</h3>
            </div>
            <pre className="rounded-lg bg-muted/50 border border-border/40 p-4 text-xs font-mono text-muted-foreground overflow-x-auto">
              <code>{`git clone https://github.com/SK8-infi/SABER
cd SABER
python -m venv .venv
.venv\\Scripts\\activate
pip install -r Saber/requirements.txt`}</code>
            </pre>
          </div>

          {/* Run Server Card */}
          <div className="rounded-2xl border border-border/60 bg-card/60 p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-emerald-500/10 border border-emerald-500/30 p-2.5">
                <Terminal className="size-5 text-emerald-400" />
              </div>
              <h3 className="font-semibold text-base">2. Launch Dashboard</h3>
            </div>
            <pre className="rounded-lg bg-muted/50 border border-border/40 p-4 text-xs font-mono text-muted-foreground overflow-x-auto">
              <code>{`# Start the SABER retrieval API
uvicorn Saber.server:app --port 8000

# Start the web dashboard
cd newFrontend && npx next dev -p 3000`}</code>
            </pre>
          </div>

          {/* System Requirements Card */}
          <div className="rounded-2xl border border-border/60 bg-card/60 p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-sky-500/10 border border-sky-500/30 p-2.5">
                <Cpu className="size-5 text-sky-400" />
              </div>
              <h3 className="font-semibold text-base">System Requirements</h3>
            </div>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-[#FBBA72] inline-block" />
                NVIDIA GPU with CUDA (≥ 4GB VRAM)
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-[#FBBA72] inline-block" />
                Python 3.10+ with PyTorch 2.x
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-[#FBBA72] inline-block" />
                Node.js 18+ for the web dashboard
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-emerald-400 inline-block" />
                Peak VRAM: 918 MB (inference only)
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-emerald-400 inline-block" />
                Zero-GPU fast path via pre-computed FAISS index
              </li>
            </ul>
          </div>

          {/* Dataset Card */}
          <div className="rounded-2xl border border-border/60 bg-card/60 p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-violet-500/10 border border-violet-500/30 p-2.5">
                <Database className="size-5 text-violet-400" />
              </div>
              <h3 className="font-semibold text-base">Dataset & Checkpoints</h3>
            </div>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-violet-400 inline-block" />
                BEN-14K: 14,832 paired S1/S2 scenes
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-violet-400 inline-block" />
                19-class BigEarthNet multi-hot labels
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-[#FBBA72] inline-block" />
                Pre-trained encoder + CFM bridge checkpoints
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-[#FBBA72] inline-block" />
                Pre-computed FAISS index (saber_search_db.pth)
              </li>
              <li className="flex items-center gap-2">
                <span className="size-1.5 rounded-full bg-emerald-400 inline-block" />
                <a
                  href={siteConfig.links.github}
                  className="underline underline-offset-2 hover:text-foreground transition-colors"
                  target="_blank"
                  rel="noreferrer"
                >
                  Download from GitHub Releases →
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </Section>
  );
}
