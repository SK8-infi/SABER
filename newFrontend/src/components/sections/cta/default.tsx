import { cn } from "@/lib/utils";
import Glow from "../../ui/glow";
import { LinkButton } from "../../ui/link-button";
import { Section } from "../../ui/section";

export default function CTA({ className }: { className?: string }) {
  return (
    <Section className={cn("group relative overflow-hidden", className)}>
      <div className="max-w-container relative z-10 mx-auto flex flex-col items-center gap-6 text-center sm:gap-8">
        <span className="text-xs font-bold uppercase tracking-widest text-[#FBBA72]">
          ISRO BAH 2026 · Problem Statement 11
        </span>
        <h2 className="max-w-[640px] text-3xl leading-tight font-semibold sm:text-5xl sm:leading-tight">
          Try SABER live — real satellite data, real retrieval
        </h2>
        <p className="text-muted-foreground max-w-[480px] text-base leading-relaxed">
          Query SAR imagery against a 14,832-scene Sentinel gallery. See the CFM bridge in action. Compare SABER vs ISRO baseline side by side.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <LinkButton variant="default" size="lg" href="/dashboard/format/query">
            Launch Demo
          </LinkButton>
          <LinkButton variant="glow" size="lg" href="https://github.com/SK8-infi/SABER">
            View on GitHub
          </LinkButton>
        </div>
      </div>
      <div className="absolute top-0 left-0 h-full w-full translate-y-[1rem] opacity-80 transition-all duration-500 ease-in-out group-hover:translate-y-[-2rem] group-hover:opacity-100">
        <Glow variant="bottom" />
      </div>
    </Section>
  );
}
