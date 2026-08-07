import {
  BlocksIcon,
  EclipseIcon,
  FastForwardIcon,
  LanguagesIcon,
  MonitorSmartphoneIcon,
  RocketIcon,
  ScanFaceIcon,
  SquarePenIcon,
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
  title?: ReactNode;
  items?: ItemProps[] | false;
  className?: string;
}

const DEFAULT_ITEMS: ItemProps[] = [
  {
    title: "Cross-Modal CFM Bridge",
    description: "Continuous Flow Matching probability ODE translating SAR to Optical latent space.",
    icon: <FastForwardIcon className="size-5 stroke-1" />,
  },
  {
    title: "Wavelength-Conditioned ViT",
    description: "Frozen DOFA backbone with dynamic central wavelength patch hypernetworks.",
    icon: <BlocksIcon className="size-5 stroke-1" />,
  },
  {
    title: "Sub-30ms FAISS Search",
    description: "Real-time vector similarity indexing with k-reciprocal graph re-ranking.",
    icon: <RocketIcon className="size-5 stroke-1" />,
  },
  {
    title: "Multi-Sensor Support",
    description: "Sentinel-1 SAR, Sentinel-2 Optical, Gaofen-1 PAN, and Gaofen-1 MS satellite data.",
    icon: <MonitorSmartphoneIcon className="size-5 stroke-1" />,
  },
  {
    title: "Multi-Label Jaccard Loss",
    description: "Direct alignment of cosine embedding geometry with ground-truth land cover overlap.",
    icon: <ScanFaceIcon className="size-5 stroke-1" />,
  },
  {
    title: "LoRA Parameter Efficiency",
    description: "Only 1.82% trainable parameters (2.06M) for low-overhead fine-tuning.",
    icon: <EclipseIcon className="size-5 stroke-1" />,
  },
  {
    title: "Telemetry Latency Profiling",
    description: "Nanosecond timing breakdown across feature extraction, ODE flow, and FAISS indexing.",
    icon: <LanguagesIcon className="size-5 stroke-1" />,
  },
  {
    title: "Multi-Sensor Inspector",
    description: "Interactive candidate scene inspector analyzing spectral channels and class overlaps.",
    icon: <SquarePenIcon className="size-5 stroke-1" />,
  },
];

export default function Items({
  title = (
    <>
      SABER Core Architecture <br className="hidden sm:inline" />
      & Capabilities
    </>
  ),
  items = DEFAULT_ITEMS,
  className,
}: ItemsProps) {
  return (
    <Section className={className}>
      <div className="max-w-container mx-auto flex flex-col items-center gap-6 sm:gap-20">
        <h2 className="max-w-[700px] text-center text-3xl leading-tight font-semibold sm:text-5xl sm:leading-tight">
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
