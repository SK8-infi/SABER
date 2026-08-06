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

function formatToThousands(value: number) {
  return Math.round(value / 100) / 10;
}

const DEFAULT_STATS: StatItemProps[] = [
  {
    label: "gallery scenes",
    value: "14.8k",
    description: "real BEN-14K Sentinel-1/2 paired scenes",
  },
  {
    label: "cross-modal",
    value: "83.4",
    suffix: "%",
    description: "F1@5 on BEN-14K — 11% above ISRO baseline",
  },
  {
    label: "retrieval in",
    value: 28,
    suffix: "ms",
    description: "end-to-end latency including CFM bridge on CPU",
  },
  {
    label: "only",
    value: "0.26",
    suffix: "%",
    description: "trainable parameters — 294.9K of 111.6M total",
  },
];

export default function Stats({
  items = DEFAULT_STATS,
  className,
}: StatsProps) {
  return (
    <Section className={className}>
      <div className="container mx-auto max-w-[960px]">
        {items !== false && items.length > 0 && (
          <div className="grid grid-cols-2 gap-12 sm:grid-cols-4">
            {items.map((item) => (
              <div
                key={`${item.label}-${item.description}`}
                className="flex flex-col items-start gap-3 text-left"
              >
                {item.label && (
                  <div className="text-muted-foreground text-sm font-semibold">
                    {item.label}
                  </div>
                )}
                <div className="flex items-baseline gap-2">
                  <div className="from-foreground to-foreground dark:to-brand bg-linear-to-r bg-clip-text text-4xl font-medium text-transparent drop-shadow-[2px_1px_24px_var(--brand-foreground)] transition-all duration-300 sm:text-5xl md:text-6xl">
                    {item.value}
                  </div>
                  {item.suffix && (
                    <div className="text-brand text-2xl font-semibold">
                      {item.suffix}
                    </div>
                  )}
                </div>
                {item.description && (
                  <div className="text-muted-foreground text-sm font-semibold text-pretty">
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
