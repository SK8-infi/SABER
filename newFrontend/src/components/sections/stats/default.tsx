import { siteConfig } from "@/config/site";

import { Section } from "../../ui/section";

interface StatItemProps {
  label?: string;
  value: string | number;
  suffix?: string;
  description?: string;
}

interface StatsProps {
  items?: StatItemProps[] | false;
  className?: string;
}

const DEFAULT_STATS: StatItemProps[] = [
  {
    label: "retrieval accuracy",
    value: "73.5%",
    description: "Cross-Modal F1@5 score (S1 SAR → S2 Optical)",
  },
  {
    label: "rank precision",
    value: "91.5%",
    description: "Cross-Modal mAP (Mean Average Precision)",
  },
  {
    label: "query latency",
    value: "<28.5",
    suffix: "ms",
    description: "Total end-to-end multi-sensor retrieval time",
  },
  {
    label: "parameter footprint",
    value: "0.26%",
    description: "Trainable LoRA parameter ratio (294.9K params)",
  },
];

export default function Stats({
  items = DEFAULT_STATS,
  className,
}: StatsProps) {
  return (
    <Section id="results" className={className}>
      <div className="container mx-auto max-w-[960px]">
        {items !== false && items.length > 0 && (
          <div className="grid grid-cols-2 gap-12 sm:grid-cols-4">
            {items.map((item) => (
              <div
                key={`${item.label}-${item.description}`}
                className="flex flex-col items-start gap-3 text-left font-sans"
              >
                {item.label && (
                  <div className="text-muted-foreground text-sm font-semibold uppercase tracking-wider">
                    {item.label}
                  </div>
                )}
                <div className="flex items-baseline gap-2">
                  <div className="from-foreground via-brand-foreground to-foreground bg-linear-to-r bg-clip-text text-4xl font-extrabold text-transparent drop-shadow-[2px_1px_24px_var(--brand-foreground)] transition-all duration-300 sm:text-5xl md:text-6xl tracking-tight">
                    {item.value}
                  </div>
                  {item.suffix && (
                    <div className="text-brand text-2xl font-bold">
                      {item.suffix}
                    </div>
                  )}
                </div>
                {item.description && (
                  <div className="text-muted-foreground text-xs sm:text-sm font-medium text-pretty">
                    {item.description}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </Section>
  );
}
