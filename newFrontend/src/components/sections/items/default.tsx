import {
  CloudOff,
  Zap,
  Layers,
  BrainCircuit,
  SatelliteIcon,
  GitMerge,
  SearchCheck,
  ShieldCheck,
} from "lucide-react";
import { ReactNode } from "react";

import { Item, ItemDescription, ItemIcon, ItemTitle } from "../../ui/item";
import { Section } from "../../ui/section";

interface ItemProps {
  title: string;
  description: string;
  icon: ReactNode;
}

interface ItemsProps {
  title?: string;
  items?: ItemProps[] | false;
  className?: string;
}

const DEFAULT_ITEMS: ItemProps[] = [
  {
    title: "All-Weather Retrieval",
    description: "SAR radar penetrates clouds, smoke, and darkness — SABER queries anytime, any condition",
    icon: <CloudOff className="size-5 stroke-1" />,
  },
  {
    title: "Sub-30ms Latency",
    description: "FAISS flat cosine index with 768-dim embeddings delivers results in under 30 milliseconds",
    icon: <Zap className="size-5 stroke-1" />,
  },
  {
    title: "CFM Latent Bridge",
    description: "Continuous Flow Matching ODE transports SAR embeddings to the optical hypersphere",
    icon: <GitMerge className="size-5 stroke-1" />,
  },
  {
    title: "DOFA ViT Backbone",
    description: "Wavelength-conditioned Vision Transformer pretrained on 100+ Earth observation bands",
    icon: <BrainCircuit className="size-5 stroke-1" />,
  },
  {
    title: "Dual Sensor Support",
    description: "Sentinel-1 SAR (2ch) ↔ Sentinel-2 MS (12ch) and Gaofen-1 PAN/MS supported",
    icon: <SatelliteIcon className="size-5 stroke-1" />,
  },
  {
    title: "Jaccard Re-ranking",
    description: "Semantic land-cover overlap re-ranking boosts precision at top-5 by 6.2%",
    icon: <SearchCheck className="size-5 stroke-1" />,
  },
  {
    title: "Multi-Modal Embeddings",
    description: "Shared 768-dim hypersphere aligns heterogeneous sensors into one retrieval space",
    icon: <Layers className="size-5 stroke-1" />,
  },
  {
    title: "ISRO PS-11 Compliant",
    description: "Evaluated on real non-synthetic partitions — 20% query / 80% gallery split",
    icon: <ShieldCheck className="size-5 stroke-1" />,
  },
];

export default function Items({
  title = "Built for real Earth observation.",
  items = DEFAULT_ITEMS,
  className,
}: ItemsProps) {
  return (
    <Section className={className}>
      <div className="max-w-container mx-auto flex flex-col items-center gap-6 sm:gap-20">
        <h2 className="max-w-[560px] text-center text-3xl leading-tight font-semibold sm:text-5xl sm:leading-tight">
          {title}
        </h2>
        {items !== false && items.length > 0 && (
          <div className="grid auto-rows-fr grid-cols-2 gap-0 sm:grid-cols-3 sm:gap-4 lg:grid-cols-4">
            {items.map((item) => (
              <Item key={item.title}>
                <ItemTitle className="flex items-center gap-2">
                  <ItemIcon>{item.icon}</ItemIcon>
                  {item.title}
                </ItemTitle>
                <ItemDescription>{item.description}</ItemDescription>
              </Item>
            ))}
          </div>
        )}
      </div>
    </Section>
  );
}
